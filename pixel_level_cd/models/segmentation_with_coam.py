import torch
import torch.nn as nn
from loguru import logger as L
import pytorch_lightning as pl
from torchvision.models.detection.image_list import ImageList
import kornia as K

from models.coattention import CoAttentionModule
from models.unet import Unet

from segmentation_models_pytorch.base import SegmentationHead
from torchgeometry.losses.dice import DiceLoss
from models.metrics import SegmentationMetrics
from pytorch_lightning.utilities import rank_zero_only
import wandb
from data.datamodule import DataModule


class SegmentationWithCoAttention(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.classes = 1  # future support for multiple classes
        self.weight_decay = args.weight_decay
        self.lr_scheduler = args.lr_scheduler if 'lr_scheduler' in args else None
        self.loss = args.loss if ('loss' in args) else ""
        number_of_coam_layers, coam_input_channels, coam_hidden_channels = args.coam_layer_data
        self.unet_decoder_channels = args.unet_decoder_channels if ('unet_decoder_channels' in args) else (
            256, 256, 128, 128, 64)
        self.unet_model = Unet(
            args.encoder,
            decoder_channels=self.unet_decoder_channels,
            encoder_depth=5,
            encoder_weights="imagenet",
            num_coam_layers=number_of_coam_layers,
            decoder_attention_type=args.decoder_attention,
            disable_segmentation_head=False,
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
        segmentation_head_class = eval(args.segmentation_head["class"]) if (
                    'segmentation_head' in args) else SegmentationHead
        print('segmentation_head_type', type(segmentation_head_class))
        self.segmentation_head = segmentation_head_class(in_channels=self.unet_decoder_channels[-1],
                                                         out_channels=self.classes)
        if args.load_weights_from is not None:
            try:
                self.safely_load_state_dict(torch.load(args.load_weights_from)["state_dict"])
            except:
                self.safely_load_state_dict(
                    torch.load(args.load_weights_from, map_location=torch.device('cpu'))["state_dict"])

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
        parser = parent_parser.add_argument_group("SegmentationWithCoAttention")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--encoder", type=str, choices=["resnet50", "resnet18"])
        parser.add_argument("--coam_layer_data", nargs="+", type=int)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--decoder_attention", type=str, default=None)
        return parent_parser

    def training_step(self, batch, batch_idx):
        batch['left_bit_mask'].div_(255)
        batch['right_bit_mask'].div_(255)

        left_image_outputs, right_image_outputs = self(batch)

        left_losses = nn.BCELoss()(left_image_outputs.squeeze(1),
                                   batch['left_bit_mask'])
        right_losses = nn.BCELoss()(right_image_outputs.squeeze(1),
                                    batch['right_bit_mask'])

        # left_losses = DiceLoss()(left_image_outputs,
        #                          batch['left_bit_mask'].to(torch.int64))
        # right_losses = DiceLoss()(right_image_outputs,
        #                           batch['right_bit_mask'].to(torch.int64))

        overall_loss = left_losses + right_losses
        self.log("train/left_losses", left_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/right_losses", right_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        left_pred_bit_mask = torch.argmax(left_image_outputs, dim=1)  # useful mainly for multi-class segmentation
        right_pred_bit_mask = torch.argmax(right_image_outputs, dim=1)

        # return overall_loss
        return {'loss': overall_loss, 'left_pred_mask': left_pred_bit_mask, 'right_pred_mask': right_pred_bit_mask}

    def validation_step(self, batch, batch_index):
        batch['left_bit_mask'].div_(255)
        batch['right_bit_mask'].div_(255)

        left_image_outputs, right_image_outputs = self(batch)

        left_losses = nn.BCELoss()(left_image_outputs.squeeze(1),
                                   batch['left_bit_mask'])
        right_losses = nn.BCELoss()(right_image_outputs.squeeze(1),
                                    batch['right_bit_mask'])

        # left_losses = DiceLoss()(left_image_outputs,
        #                          batch['left_bit_mask'].to(torch.int64))
        # right_losses = DiceLoss()(right_image_outputs,
        #                           batch['right_bit_mask'].to(torch.int64))

        overall_loss = left_losses + right_losses
        self.log("val/left_losses", left_losses, on_epoch=True)
        self.log("val/right_losses", right_losses, on_epoch=True)
        self.log("val/overall_loss", overall_loss, on_epoch=True)
        self.log("val_loss", overall_loss, on_epoch=True)

        truth = torch.cat([batch['left_bit_mask'], batch['right_bit_mask']], dim=-2)
        pred = torch.cat([left_image_outputs, right_image_outputs], dim=-2)
        pixel_acc, dice, precision, recall = SegmentationMetrics()(truth, pred)
        self.log("val/pixel_acc", pixel_acc, on_epoch=True)
        self.log("val/dice", dice, on_epoch=True)
        self.log("val/precision", precision, on_epoch=True)
        self.log("val/recall", recall, on_epoch=True)

        left_pred_bit_mask = torch.argmax(left_image_outputs, dim=1)
        right_pred_bit_mask = torch.argmax(right_image_outputs, dim=1)

        return {'loss': overall_loss, 'left_pred_mask': left_pred_bit_mask, 'right_pred_mask': right_pred_bit_mask}

    def test_step(self, batch, batch_index, dataloader_index=0):
        batch['left_bit_mask'].div_(255)
        batch['right_bit_mask'].div_(255)
        left_image_outputs, right_image_outputs = self(batch)

        left_losses = nn.BCELoss()(left_image_outputs.squeeze(1),
                                   batch['left_bit_mask'])
        right_losses = nn.BCELoss()(right_image_outputs.squeeze(1),
                                    batch['right_bit_mask'])

        # left_losses = DiceLoss()(left_image_outputs,
        #                          batch['left_bit_mask'].to(torch.int64))
        # right_losses = DiceLoss()(right_image_outputs,
        #                           batch['right_bit_mask'].to(torch.int64))

        overall_loss = left_losses + right_losses
        self.log("test/left_losses", left_losses, on_epoch=True)
        self.log("test/right_losses", right_losses, on_epoch=True)
        self.log("test/overall_loss", overall_loss, on_epoch=True)

        truth = torch.cat([batch['left_bit_mask'], batch['right_bit_mask']], dim=-2)
        pred = torch.cat([left_image_outputs, right_image_outputs], dim=-2)
        pixel_acc, dice, precision, recall = SegmentationMetrics()(truth, pred)
        self.log("test/pixel_acc", pixel_acc, on_epoch=True)
        self.log("test/dice", dice, on_epoch=True)
        self.log("test/precision", precision, on_epoch=True)
        self.log("test/recall", recall, on_epoch=True)

        left_pred_bit_mask = torch.argmax(left_image_outputs, dim=1)
        right_pred_bit_mask = torch.argmax(right_image_outputs, dim=1)

        return {'loss': overall_loss, 'left_pred_mask': left_pred_bit_mask, 'right_pred_mask': right_pred_bit_mask}

    def validation_epoch_end(self, val_set_outputs):
        val_loss = 0
        for x in val_set_outputs:
            val_loss += x['loss']
        val_loss = val_loss / len(val_set_outputs)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)

    # def predict(self, batch, batch_index):
    #     left_image_outputs, right_image_outputs = self(batch)
    #     return left_image_outputs, right_image_outputs

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

        if self.segmentation_head is not None:
            left_image_pred_mask = self.segmentation_head(left_image_decoded_features)
            right_image_pred_mask = self.segmentation_head(right_image_decoded_features)
            left_image_pred_mask = nn.Sigmoid()(left_image_pred_mask)
            right_image_pred_mask = nn.Sigmoid()(right_image_pred_mask)
        else:
            left_image_pred_mask = left_image_decoded_features
            right_image_pred_mask = right_image_decoded_features

        return left_image_pred_mask, right_image_pred_mask


def marshal_getitem_data(data, split):
    """
    The data field above is returned by the individual datasets.
    This function marshals that data into the format expected by this
    model/method.
    """
    return {
        "left_image": data["image1"],
        "right_image": data["image2"],
        "left_bit_mask": data["label1"],
        "right_bit_mask": data["label2"],
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
    return collated_dictionary


class WandbCallbackManager(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        datamodule = DataModule(args)
        datamodule.setup()
        # self.test_set_names = datamodule.test_dataset_names
        self.train_batch = None
        self.train_predicted_masks = None
        self.train_target_masks = None

    @rank_zero_only
    def on_fit_start(self, trainer, model):
        if self.args.no_logging:
            return
        trainer.logger.experiment.config.update(self.args, allow_val_change=True)

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, model, output, batch, batch_idx, dataloader_idx=0
    ):
        self.val_batch = batch
        self.val_predicted_masks = (output['left_pred_mask'].to('cpu'),
                                    output['right_pred_mask'].to('cpu'))
        self.val_target_masks = batch['left_bit_mask'].to('cpu'), batch['right_bit_mask'].to('cpu')
        # if batch_idx == 0:

    @rank_zero_only
    def on_validation_end(self, trainer, model):
        self.log_qualitative_predictions(
            self.val_batch,
            self.val_predicted_masks,
            self.val_target_masks,
            "val",
            trainer,
        )

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, model, output, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self.test_batch = batch
            self.test_predicted_masks = (output['left_pred_mask'].to('cpu'),
                                          output['right_pred_mask'].to('cpu'))
            self.test_target_masks = batch['left_bit_mask'].to('cpu'), batch['right_bit_mask'].to('cpu')

    @rank_zero_only
    def on_test_end(self, trainer, model):
        self.log_qualitative_predictions(
            self.test_batch,
            self.test_predicted_masks,
            self.test_target_masks,
            "test",
            trainer,
        )

    def on_train_batch_end(self, trainer, model, output, batch, batch_idx, dataloader_idx=0):
        if batch_idx == trainer.num_training_batches-1:
            self.train_batch = batch
            self.train_predicted_masks = (output['left_pred_mask'].to('cpu'),
                                          output['right_pred_mask'].to('cpu'))
            self.train_target_masks = batch['left_bit_mask'].to('cpu'), batch['right_bit_mask'].to('cpu')

    def on_train_epoch_end(self, trainer, model):
        self.log_qualitative_predictions(
            self.train_batch,
            self.train_predicted_masks,
            self.train_target_masks,
            "train",
            trainer,
        )

    def log_qualitative_predictions(
        self,
        batch,
        predicted_masks,
        target_masks,
        batch_name,
        trainer,
    ):
        """
        Logs the predicted masks for a single val/test batch for qualitative analysis.
        """
        if (batch is None) or (predicted_masks is None) or (target_masks is None):
            return

        outputs = []
        # print('K.tensor_to_image(batch["left_image"])', K.tensor_to_image(batch["left_image"]).shape)
        left_pred_masks, right_pred_masks = predicted_masks
        left_true_masks, right_true_masks = target_masks
        for i in range(min(5, batch['left_image'].shape[0])):
            # outputs.append(wandb.Image(
            #                       K.tensor_to_image(batch["left_image"][i].squeeze().to('cpu')),
            #                       masks={
            #                             "predictions": {
            #                                 "mask_data": left_pred_masks[i].squeeze().to('cpu').detach().numpy()
            #                             },
            #                             "ground_truth": {
            #                                 "mask_data": left_true_masks[i].squeeze().to('cpu').detach().numpy()
            #                             }
            #                 }))
            # outputs.append(wandb.Image(
            #                       K.tensor_to_image(batch["right_image"][i].squeeze().to('cpu')),
            #                       masks={
            #                             "predictions": {
            #                                 "mask_data": right_pred_masks[i].squeeze().to('cpu').detach().numpy()
            #                             },
            #                             "ground_truth": {
            #                                 "mask_data": right_true_masks[i].squeeze().to('cpu').detach().numpy()
            #                             }
            #                 }))
            outputs.append(wandb.Image(K.tensor_to_image(batch['left_image'][i].squeeze().to('cpu'))))
            outputs.append(wandb.Image(K.tensor_to_image(batch['right_image'][i].squeeze().to('cpu'))))
            outputs.append(wandb.Image(left_true_masks[i].squeeze().to('cpu').detach().numpy()))
            outputs.append(wandb.Image(left_pred_masks[i].squeeze().to('cpu').detach().numpy()))
            outputs.append(wandb.Image(right_pred_masks[i].squeeze().to('cpu').detach().numpy()))
        L.log("INFO", f"Finished computing qualitative predictions for {batch_name}.")
        if not self.args.no_logging:
            trainer.logger.experiment.log(
                {
                    f"qualitative_predictions/{batch_name}": outputs,
                    "global_step": trainer.global_step,
                }
            )