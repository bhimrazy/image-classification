from torch.utils.data import DataLoader
from src.dataset import ImageDataset
from src.utils import load_data

from src.transforms import train_transform, test_transform


def create_data_loader(
    csv_path: str,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 2,
    train_mode: bool = True,
) -> DataLoader:
    """
    Create a DataLoader object from a CSV file.

    Parameters:
    csv_path (str): The path to the CSV file.
    batch_size (int): The batch size.
    pin_memory (bool): Whether to pin memory or not.
    num_workers (int): The number of workers.
    train_mode (bool): Whether to use the dataset in training mode or not.
    """
    images, labels = load_data(csv_path)
    if images is None or labels is None:
        raise ValueError("Failed to load data.")

    transform = train_transform if train_mode else test_transform
    dataset = ImageDataset(images, labels, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=train_mode,
    )
