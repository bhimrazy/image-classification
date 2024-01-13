import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
    argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model.")
    #  train csv path
    parser.add_argument(
        "--train-csv-path",
        type=str,
        required=True,
        help="The path to the training CSV file.",
    )
    #  test csv path
    parser.add_argument(
        "--test-csv-path",
        type=str,
        required=True,
        help="The path to the testing CSV file.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="The learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="The interval for logging.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Whether to save the model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="The number of workers.",
    )
    return parser.parse_args()
