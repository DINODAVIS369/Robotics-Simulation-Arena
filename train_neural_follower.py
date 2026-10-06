import argparse
from pathlib import Path

import numpy as np

from Controller.neural_follower import (
    DEFAULT_WEIGHTS_PATH,
    generate_imitation_data,
    initialize_network,
    predict_network,
    save_network,
    train_network,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train the study neural follower to imitate Regulated Pure Pursuit."
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    features, labels = generate_imitation_data(
        sample_count=args.samples,
        seed=args.seed,
    )
    split = max(1, int(len(features) * 0.8))
    training_features, validation_features = features[:split], features[split:]
    training_labels, validation_labels = labels[:split], labels[split:]
    network = initialize_network(seed=args.seed + 1)
    initial_loss = float(
        np.mean((predict_network(network, validation_features) - validation_labels) ** 2)
    )
    training_loss = train_network(
        network,
        training_features,
        training_labels,
        epochs=args.epochs,
    )
    validation_loss = float(
        np.mean((predict_network(network, validation_features) - validation_labels) ** 2)
    )
    save_network(network, args.output)

    print(f"training samples: {len(training_features)}")
    print(f"validation samples: {len(validation_features)}")
    print(f"validation MSE before training: {initial_loss:.6f}")
    print(f"training MSE after training: {training_loss:.6f}")
    print(f"validation MSE after training: {validation_loss:.6f}")
    print(f"saved model: {args.output}")


if __name__ == "__main__":
    main()
