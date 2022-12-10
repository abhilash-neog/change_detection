import os
import pickle

import numpy as np

from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from utils.general import cache_data_triton

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


class Levir(Dataset):
    def __init__(self, path_to_dataset, split, method, image_size, image_transformation=None, machine="local"):
        self.path_to_dataset = path_to_dataset
        train_val_test_split = self.get_train_val_test_split(split)
        self.indices = train_val_test_split[split]
        self.split = split
        self.image_size = image_size
        self.machine = machine
        # self.image_augmentations = AugmentationPipeline(
        #     mode=split, path_to_dataset=path_to_dataset, image_transformation=image_transformation
        # )
        self.marshal_getitem_data = self.import_method_specific_functions(method)

    def import_method_specific_functions(self, method):
        if method == "segmentation":
            from models.segmentation_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def get_train_val_test_split(self, split):
        print(f"Path to dataset = {self.path_to_dataset}")
        train_val_test_split_file_path = os.path.join(self.path_to_dataset, "data_split.pkl")
        if os.path.exists(train_val_test_split_file_path):
            with open(train_val_test_split_file_path, "rb") as file:
                return pickle.load(file)

        with open(os.path.join(self.path_to_dataset, "image_ids.txt")) as f:
            image_ids = f.readlines()

        image_ids = [f.strip() for f in image_ids]

        np.random.shuffle(image_ids)
        if split == "test":
            train_val_test_split = {
                "test": image_ids,
            }
        else:
            number_of_images = len(image_ids)
            number_of_train_images = int(0.95 * number_of_images)
            train_val_test_split = {
                "train": image_ids[:number_of_train_images],
                "val": image_ids[number_of_train_images:],
            }
        with open(train_val_test_split_file_path, "wb") as file:
            pickle.dump(train_val_test_split, file)
        return train_val_test_split

    def read_image_as_tensor(self, path_to_image):
        """
        Returns a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size))
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def read_bitmask_as_tensor(self, path_to_image):
        """
        Returns a segmentation bitmask image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("L")
        pil_image = pil_image.resize((self.image_size, self.image_size))
        pil_image = pil_image.point(lambda x: 255 if x < 255 else 0, "1")
        image_as_tensor = pil_to_tensor(pil_image).float()

        return image_as_tensor

    def __len__(self):
        """
        Returns the number of training/validation/testing images.
        """
        return len(self.indices)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        img_id = self.indices[item_index]

        image_1 = self.read_image_as_tensor(
            cache_data_triton(self.path_to_dataset, os.path.join('A', img_id), self.machine))
        image_2 = self.read_image_as_tensor(
            cache_data_triton(self.path_to_dataset, os.path.join('B', img_id), self.machine))
        label_1 = self.read_bitmask_as_tensor(
            cache_data_triton(self.path_to_dataset, os.path.join('label_1', img_id), self.machine))
        label_2 = self.read_bitmask_as_tensor(
            cache_data_triton(self.path_to_dataset, os.path.join('label_2', img_id), self.machine))

        # (image_1, image_2, label_1, label_2) = self.image_augmentations(
        #     image_1,
        #     image_2,
        #     label_1,
        #     label_2,
        #     item_index,
        # )

        return {
            "image1": image_1.squeeze(),
            "image2": image_2.squeeze(),
            "label1": label_1.squeeze(),
            "label2": label_2.squeeze(),
        }
