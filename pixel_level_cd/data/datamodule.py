import numpy as np
import pytorch_lightning as pl
from loguru import logger as L
from torch.utils.data import ConcatDataset, DataLoader

from data.levir import Levir


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_dataloader_workers = args.num_dataloader_workers
        self.method = args.method
        self.dataset_configs = args.datasets
        self.dataloader_collate_fn = self.import_method_specific_functions(self.method)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LevirDataModule")
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--num_dataloader_workers", type=int)
        parser.add_argument("--test_batch_size", type=int)
        return parent_parser

    def import_method_specific_functions(self, method):
        if method == "segmentation":
            from models.segmentation_with_coam import dataloader_collate_fn
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return dataloader_collate_fn

    def collate_fn(self, batch, dataset):
        """
        A wrapper collate function that calls method-specific,
        data collation functions. It also takes care of filtering out
        any None batch items and if the batch ends up empty, it attempts
        to create a batch of a single non-None item.
        """
        batch = [x for x in batch if x is not None]
        tries = 0
        while len(batch) == 0:
            tries += 1
            random_item = dataset[np.random.randint(0, len(dataset))]
            if random_item is not None:
                batch = [random_item]
            if tries % 50 == 0:
                L.log(
                    "DEBUG",
                    f"Made {tries} attempts to construct a non-None batch.\
                        If this happens too often, maybe it's not a good workaround.",
                )
        return self.dataloader_collate_fn(batch)

    def setup(self, stage=None):
        train_dataset_config = self.dataset_configs["train_dataset"]
        self.train_dataset = eval(train_dataset_config["class"])(**train_dataset_config["args"])

        val_dataset_config = self.dataset_configs["val_dataset"]

        self.val_dataset = eval(val_dataset_config["class"])(**val_dataset_config["args"])
        test_dataset_configs = self.dataset_configs["test_datasets"]

        self.test_dataset = eval(test_dataset_configs["class"])(**test_dataset_configs["args"])

    def train_dataloader(self):
        def collate_fn_wrapper(batch):
            return self.collate_fn(batch, self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            # collate_fn=collate_fn_wrapper,
        )

    def val_dataloader(self):
        def collate_fn_wrapper(batch):
            return self.collate_fn(batch, self.val_dataset)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            # collate_fn=collate_fn_wrapper,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers
        )
