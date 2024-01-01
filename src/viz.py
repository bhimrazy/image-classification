import matplotlib.pyplot as plt
from typing import Tuple


def plot_data_distribution(
    labels: list,
) -> None:
    """
    Plot the data distribution.

    Parameters:
    labels (list): The labels.
    labels_names (list): The labels names.
    """
    if not isinstance(labels, list):
        raise ValueError("labels must be a list.")

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.bar(labels)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("artifacts/data_distribution.png")
    plt.close()


def plot_results(
    results: dict,
    loss_keys: Tuple[str, str] = ("train_loss", "test_loss"),
    acc_keys: Tuple[str, str] = ("train_acc", "test_acc"),
) -> None:
    """
    Plot the results.

    Parameters:
    results (dict): The results.
    loss_keys (Tuple[str, str]): The keys to plot for loss.
    acc_keys (Tuple[str, str]): The keys to plot for accuracy.
    """
    if not isinstance(results, dict):
        raise ValueError("results must be a dictionary.")
    if not all(key in results for key in (*loss_keys, *acc_keys)):
        raise ValueError(f"results must contain the keys: {loss_keys} and {acc_keys}")

    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    for ax, keys, ylabel in zip(axs, (loss_keys, acc_keys), ("Loss", "Accuracy")):
        for key in keys:
            ax.plot(results[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("artifacts/results.png")
    plt.close()


if __name__ == "__main__":
    results = {
        "train_loss": [1.0, 0.5, 0.2, 0.1],
        "train_acc": [0.5, 0.6, 0.7, 0.8],
        "test_loss": [1.0, 0.5, 0.2, 0.1],
        "test_acc": [0.5, 0.6, 0.7, 0.8],
    }
    plot_results(results)
