import os
from datetime import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data_loader import create_data_loader
from src.model import DenseNetModel
from src.parser import parse_args
from src.utils import get_device, seed_everything, train_eval_step
from src.viz import plot_results

runs_dir = 'runs'
os.makedirs(runs_dir, exist_ok=True)



def main() -> None:
    """
    Train a model.
    """
    args = parse_args()

    # Seed everything for reproducibility
    seed_everything(args.seed)

    # Get the device to use for training
    device = get_device()
    print(f"Using device: {device}")

    # Create a DataLoader object for training and testing
    print("Creating data loaders...")
    train_loader = create_data_loader(
        args.train_csv_path,
        args.batch_size,
        train_mode=True,
        num_workers=args.num_workers,
    )
    test_loader = create_data_loader(
        args.test_csv_path,
        args.batch_size,
        train_mode=False,
        num_workers=args.num_workers,
    )

    num_classes = len(set(train_loader.dataset.image_labels))
    print(f"Number of classes: {num_classes}")

    print("Training...")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = DenseNetModel(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs", unit="epoch"):
        train_loss, train_acc = train_eval_step(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            train_mode=True,
            log_interval=args.log_interval,
        )
        test_loss, test_acc = train_eval_step(
            model,
            test_loader,
            criterion,
            device,
            optimizer,
            train_mode=False,
            log_interval=args.log_interval,
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Train Acc: {train_acc:.2f} | "
            f"Test Loss: {test_loss:.3f} | "
            f"Test Acc: {test_acc:.2f}"
        )

    if args.save_model:
        torch.save(model.state_dict(), "model.pth")

    plot_results(results)

    # Tensorboard
    run_name = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    writer = SummaryWriter(os.path.join(runs_dir, run_name))

    # log results
    writer.add_scalars('loss', {'train': results["train_loss"], 'test': results["test_loss"]}, 0)
    writer.add_scalars('accuracy', {'train': results["train_acc"], 'test': results["test_acc"]}, 0)

    # log model
    writer.add_graph(model, train_loader.dataset[0][0].unsqueeze(0).to(device))

    # log images
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)

    # log hyperparameters
    writer.add_hparams(
        {'lr': args.lr, 'batch_size': args.batch_size, 'epochs': args.epochs},
        {'hparam/accuracy': results["test_acc"][-1], 'hparam/loss': results["test_loss"][-1]}
    )

    # close writer
    writer.close()



if __name__ == "__main__":
    main()
