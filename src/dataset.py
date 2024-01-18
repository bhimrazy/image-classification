from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Union, Tuple


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        image_labels: List[int],
        transform: transforms.Compose = None,
        label_transform=None,
        is_ulb: bool = False,
        strong_transform: transforms.Compose = None,
    ):
        """
        Custom dataset for image classification.

        Args:
            image_paths (List[str]): List of file paths to images.
            image_labels (List[int]): List of corresponding image labels.
            transform (transforms.Compose, optional): Transformations applied to the images. Defaults to None.
            is_ulb (bool, optional): Indicates whether the dataset is unlabeled. Defaults to False.
            strong_transform (transforms.Compose, optional): Strong transformations for unlabeled data. Defaults to None.
        """
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform
        self.label_transform = label_transform
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Image.Image, int], Tuple[Image.Image, Image.Image, int]]:
        """
        Get image and label for a given index.

        Args:
            idx (int): Index of the dataset.

        Returns:
            Union[Tuple[Image.Image, int], Tuple[Image.Image, Image.Image, int]]:
            Tuple containing image and label or image, strong-transformed image, and label.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.image_labels[idx]

        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.transform is not None:
            image = self.transform(image)

        if self.is_ulb and self.strong_transform is not None:
            st_image = Image.open(self.image_paths[idx]).convert("RGB")
            st_image = self.strong_transform(st_image)
            return image, st_image, label

        return image, label
