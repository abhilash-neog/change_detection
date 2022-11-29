import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from easydict import EasyDict
from loguru import logger as L

from models.coattention import CoAttentionModule
from models.unet import Unet
from utils.voc_eval import BoxList, eval_detection_voc


class PixelNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.test_set_names = [test_set.name for test_set in args.datasets.test_datasets]
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        number_of_coam_layers, coam_input_channels, coam_hidden_channels = args.coam_layer_data
        self.unet_model = Unet(
            args.encoder,
            decoder_channels=(256, 256, 128, 128, 64),
            encoder_depth=5,
            encoder_weights="imagenet",
            num_coam_layers=number_of_coam_layers,
            decoder_attention_type=args.decoder_attention,
            disable_segmentation_head=True,
        )
        self.coattention_modules = nn.ModuleList(
            [
                CoAttentionModule(
                    input_channels=coam_input_channels[i],
                    hidden_channels=coam_hidden_channels[i],
                    attention_type=args.attention,
                )
                for i in range(number_of_coam_layers)
            ]
        )

        self.centernet_head = CenterNetHead(
            in_channel=64,
            feat_channel=64,
            num_classes=1,
            test_cfg=EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100}),
        )
        self.centernet_head.init_weights()
        if args.load_weights_from is not None:
            self.safely_load_state_dict(torch.load(args.load_weights_from)["state_dict"])

    def safely_load_state_dict(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in checkpoint_state_dict:
            if k in model_state_dict:
                if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                    L.log(
                        "INFO",
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {checkpoint_state_dict[k].shape}",
                    )
                    checkpoint_state_dict[k] = model_state_dict[k]
            else:
                L.log("INFO", f"Dropping parameter {k}")
        self.load_state_dict(checkpoint_state_dict, strict=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CenterNetWithCoAttention")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--encoder", type=str, choices=["resnet50", "resnet18"])
        parser.add_argument("--coam_layer_data", nargs="+", type=int)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--decoder_attention", type=str, default=None)
        return parent_parser

    def training_step(self, batch, batch_idx):
        left_image_outputs, right_image_outputs = self(batch)
        left_losses = self.centernet_head.loss(
            *left_image_outputs,
            batch["left_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        right_losses = self.centernet_head.loss(
            *right_image_outputs,
            batch["right_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        overall_loss = 0
        for key in left_losses:
            self.log(
                f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
            )
            overall_loss += left_losses[key] + right_losses[key]
        self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
        return overall_loss

    def validation_step(self, batch, batch_index):
        left_image_outputs, right_image_outputs = self(batch)
        left_losses = self.centernet_head.loss(
            *left_image_outputs,
            batch["left_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        right_losses = self.centernet_head.loss(
            *right_image_outputs,
            batch["right_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        overall_loss = 0
        for key in left_losses:
            self.log(f"val/{key}", left_losses[key] + right_losses[key], on_epoch=True)
            overall_loss += left_losses[key] + right_losses[key]
        self.log("val/overall_loss", overall_loss, on_epoch=True)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        return left_predicted_bboxes, right_predicted_bboxes

    def test_step(self, batch, batch_index, dataloader_index=0):
        left_image_outputs, right_image_outputs = self(batch)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        return (
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in left_predicted_bboxes
            ],
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in right_predicted_bboxes
            ],
            [bboxes.cpu() for bboxes in batch["left_image_target_bboxes"]],
            [bboxes.cpu() for bboxes in batch["right_image_target_bboxes"]],
        )

    def test_epoch_end(self, multiple_test_set_outputs):
        """
        Test set evaluation function.
        """
        if len(self.test_set_names) == 1:
            multiple_test_set_outputs = [multiple_test_set_outputs]
        # iterate over all the test sets
        for test_set_name, test_set_batch_outputs in zip(
            self.test_set_names, multiple_test_set_outputs
        ):
            predicted_bboxes = []
            target_bboxes = []
            # iterate over all the batches for the current test set
            for test_set_outputs in test_set_batch_outputs:
                (
                    left_predicted_bboxes,
                    right_predicted_bboxes,
                    left_target_bboxes,
                    right_target_bboxes,
                ) = test_set_outputs
                # iterate over all predictions for images
                for bboxes_per_side in [left_predicted_bboxes, right_predicted_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        # filter out background bboxes
                        bboxes_per_image = bboxes_per_image[0][bboxes_per_image[1] == 0]
                        bbox_list = BoxList(
                            bboxes_per_image[:, :4],
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("scores", bboxes_per_image[:, 4])
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        predicted_bboxes.append(bbox_list)
                # iterate over all targets for images
                for bboxes_per_side in [left_target_bboxes, right_target_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        bbox_list = BoxList(
                            bboxes_per_image,
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        bbox_list.add_field("difficult", torch.zeros(bboxes_per_image.shape[0]))
                        target_bboxes.append(bbox_list)
            # compute metrics
            ap_map_precision_recall = eval_detection_voc(
                predicted_bboxes, target_bboxes, iou_thresh=0.5
            )
            L.log(
                "INFO",
                f"{test_set_name} AP: {ap_map_precision_recall['ap']}, mAP: {ap_map_precision_recall['map']}",
            )

    def configure_optimizers(self):
        optimizer_params = [
            {"params": [parameter for parameter in self.parameters() if parameter.requires_grad]}
        ]
        optimizer = torch.optim.Adam(optimizer_params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, batch):
        left_image_encoded_features = self.unet_model.encoder(batch["left_image"])
        right_image_encoded_features = self.unet_model.encoder(batch["right_image"])
        for i in range(len(self.coattention_modules)):
            (
                left_image_encoded_features[-(i + 1)],
                right_image_encoded_features[-(i + 1)],
            ) = self.coattention_modules[i](
                left_image_encoded_features[-(i + 1)], right_image_encoded_features[-(i + 1)]
            )
        left_image_decoded_features = self.unet_model.decoder(*left_image_encoded_features)
        right_image_decoded_features = self.unet_model.decoder(*right_image_encoded_features)
        return (
            self.centernet_head([left_image_decoded_features]),
            self.centernet_head([right_image_decoded_features]),
        )


def marshal_getitem_data(data, split):
    """
    The data field above is returned by the individual datasets.
    This function marshals that data into the format expected by this
    model/method.
    """
    if split in ["train", "val", "test"]:
        (
            data["image1"],
            target_region_and_annotations,
        ) = utils.geometry.resize_image_and_annotations(
            data["image1"],
            output_shape_as_hw=(256, 256),
            annotations=[data["image1_target_region_as_coco_annotation"]]
            + data["image1_target_annotations"],
        )
        data["image1_target_region_as_coco_annotation"] = target_region_and_annotations[0]
        data["image1_target_annotations"] = target_region_and_annotations[1:]
        (
            data["image2"],
            target_region_and_annotations,
        ) = utils.geometry.resize_image_and_annotations(
            data["image2"],
            output_shape_as_hw=(256, 256),
            annotations=[data["image2_target_region_as_coco_annotation"]]
            + data["image2_target_annotations"],
        )
        data["image2_target_region_as_coco_annotation"] = target_region_and_annotations[0]
        data["image2_target_annotations"] = target_region_and_annotations[1:]

    assert data["image1"].shape == data["image2"].shape
    image1_target_bboxes = torch.Tensor([x["bbox"] for x in data["image1_target_annotations"]])
    image2_target_bboxes = torch.Tensor([x["bbox"] for x in data["image2_target_annotations"]])

    if len(image1_target_bboxes) != len(image2_target_bboxes) or len(image1_target_bboxes) == 0:
        return None

    return {
        "left_image": data["image1"],
        "right_image": data["image2"],
        "left_image_target_bboxes": image1_target_bboxes,
        "right_image_target_bboxes": image2_target_bboxes,
        "target_bbox_labels": torch.zeros(len(image1_target_bboxes)).long(),
        "query_metadata": {
            "pad_shape": data["image1"].shape[-2:],
            "border": np.array([0, 0, 0, 0]),
            "batch_input_shape": data["image1"].shape[-2:],
        },
    }


def dataloader_collate_fn(batch):
    """
    Defines the collate function for the dataloader specific to this
    method/model.
    """
    keys = batch[0].keys()
    collated_dictionary = {}
    for key in keys:
        collated_dictionary[key] = []
        for batch_item in batch:
            collated_dictionary[key].append(batch_item[key])
        if key in [
            "left_image_target_bboxes",
            "right_image_target_bboxes",
            "query_metadata",
            "target_bbox_labels",
        ]:
            continue
        collated_dictionary[key] = ImageList.from_tensors(
            collated_dictionary[key], size_divisibility=32
        ).tensor
    collated_dictionary
    return collated_dictionary
