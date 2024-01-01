import torch
from tqdm import tqdm

from src.data_loader import create_data_loader
from src.model import DenseNetModel
from src.parser import parse_args
from src.utils import get_device, seed_everything, train_eval_step
from src.viz import plot_results


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
    )
    test_loader = create_data_loader(
        args.test_csv_path,
        args.batch_size,
        train_mode=False,
    )

    print("Training...")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = DenseNetModel(num_classes=5).to(device)

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


if __name__ == "__main__":
    main()
