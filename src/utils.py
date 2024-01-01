import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def get_device() -> torch.device:
    """
    Get the device to use for training.

    Returns:
    torch.device: The device to use for training.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def load_data(path: str) -> Tuple[List, List]:
    """
    Load image paths and labels from a CSV file.

    Parameters:
    path (str): The path to the CSV file.

    Returns:
    Tuple[List, List]: A tuple containing two pandas Series, the image paths and the labels.
    """
    try:
        df = pd.read_csv(path)
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError(
                "CSV file does not contain required columns: 'image_path' and 'label'"
            )
        images, labels = df["image_path"].values.tolist(), df["label"].values.tolist()
        return images, labels
    except FileNotFoundError:
        print(f"File {path} not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def seed_everything(seed: int = 42):
    """
    Seed everything for reproducibility.

    Parameters:
    seed (int): The random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_eval_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_mode: bool = True,
    log_interval: int = 10,
) -> Tuple[float, float]:
    """
    Perform a training or evaluation step.

    Parameters:
    model (torch.nn.Module): The model.
    data_loader (torch.utils.data.DataLoader): The data loader.
    criterion (torch.nn.Module): The loss function.
    device (torch.device): The device to use for training.
    optimizer (torch.optim.Optimizer): The optimizer.
    train_mode (bool): Whether to use the model in training mode or not.

    Returns:
    Tuple[float, float]: A tuple containing the loss and accuracy.
    """
    epoch_loss = 0.0
    epoch_corrects = 0.0

    model.train() if train_mode else model.eval()

    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            if train_mode:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.cpu().item() * images.size(0)
        epoch_corrects += torch.sum(preds == labels.data).cpu().item()

        if i % log_interval == 0:
            print(
                f"Batch {i}/{len(data_loader)} - "
                f"{'Train' if train_mode else 'Test'} Loss: {loss.item():.4f}"
            )

    epoch_loss = epoch_loss / len(data_loader.dataset)
    epoch_corrects = epoch_corrects / len(data_loader.dataset)

    return epoch_loss, epoch_corrects
