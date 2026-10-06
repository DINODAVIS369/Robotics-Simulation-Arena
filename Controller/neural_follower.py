import json
import math
from pathlib import Path

import numpy as np


INPUT_SIZE = 3
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2
MODEL_VERSION = 1
DEFAULT_WEIGHTS_PATH = Path(__file__).with_name(
    "neural_follower_weights.json"
)


def _features(local_x, local_y, goal_distance, lookahead):
    scale = max(2.0 * lookahead, 0.1)
    return np.array(
        [
            local_x / scale,
            local_y / scale,
            min(goal_distance / 3.0, 2.0),
        ],
        dtype=float,
    )


def _teacher_command(local_x, local_y, goal_distance, max_speed, max_angular_speed):
    curvature = 2.0 * local_y / max(local_x**2 + local_y**2, 1e-9)
    if abs(curvature) < 1e-9:
        curvature_speed = max_speed
    else:
        curvature_speed = min(max_speed, math.sqrt(0.6 / abs(curvature)))
    velocity = min(
        curvature_speed,
        math.sqrt(1.2 * max(goal_distance, 0.0)),
    )
    angular = max(
        -max_angular_speed,
        min(max_angular_speed, velocity * curvature),
    )
    return velocity, angular


def generate_imitation_data(
    sample_count=6000,
    seed=7,
    max_speed=0.35,
    max_angular_speed=1.5,
    lookahead=0.55,
):
    """Generate synthetic lookahead examples labelled by the RPP teacher."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    random = np.random.default_rng(seed)
    local_targets = random.uniform(-1.5, 1.5, size=(sample_count, 2))
    goal_distances = random.uniform(0.0, 4.0, size=sample_count)
    features = np.empty((sample_count, INPUT_SIZE), dtype=float)
    labels = np.empty((sample_count, OUTPUT_SIZE), dtype=float)

    for index, ((local_x, local_y), goal_distance) in enumerate(
        zip(local_targets, goal_distances)
    ):
        features[index] = _features(
            local_x,
            local_y,
            goal_distance,
            lookahead,
        )
        velocity, angular = _teacher_command(
            local_x,
            local_y,
            goal_distance,
            max_speed,
            max_angular_speed,
        )
        labels[index] = (
            velocity / max_speed,
            angular / max_angular_speed,
        )
    return features, labels


def initialize_network(seed=11):
    random = np.random.default_rng(seed)
    return {
        "w1": random.normal(
            0.0,
            math.sqrt(1.0 / INPUT_SIZE),
            size=(INPUT_SIZE, HIDDEN_SIZE),
        ),
        "b1": np.zeros(HIDDEN_SIZE, dtype=float),
        "w2": random.normal(
            0.0,
            math.sqrt(1.0 / HIDDEN_SIZE),
            size=(HIDDEN_SIZE, OUTPUT_SIZE),
        ),
        "b2": np.zeros(OUTPUT_SIZE, dtype=float),
    }


def predict_network(network, features):
    hidden = np.tanh(features @ network["w1"] + network["b1"])
    return np.tanh(hidden @ network["w2"] + network["b2"])


def train_network(network, features, labels, epochs=500, learning_rate=0.03):
    """Train a two-layer tanh network using full-batch gradient descent."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or features.shape[1] != INPUT_SIZE:
        raise ValueError(f"features must have shape (n, {INPUT_SIZE})")
    if labels.shape != (len(features), OUTPUT_SIZE):
        raise ValueError(f"labels must have shape (n, {OUTPUT_SIZE})")
    if len(features) == 0 or epochs < 1 or learning_rate <= 0:
        raise ValueError("training data, epochs, and learning rate must be positive")

    for _ in range(epochs):
        hidden = np.tanh(features @ network["w1"] + network["b1"])
        output = np.tanh(hidden @ network["w2"] + network["b2"])
        output_delta = (
            2.0
            * (output - labels)
            * (1.0 - output**2)
            / len(features)
        )
        hidden_delta = (
            output_delta @ network["w2"].T
        ) * (1.0 - hidden**2)

        network["w2"] -= learning_rate * (hidden.T @ output_delta)
        network["b2"] -= learning_rate * output_delta.sum(axis=0)
        network["w1"] -= learning_rate * (features.T @ hidden_delta)
        network["b1"] -= learning_rate * hidden_delta.sum(axis=0)

    return float(np.mean((predict_network(network, features) - labels) ** 2))


def save_network(network, path=DEFAULT_WEIGHTS_PATH):
    payload = {
        "version": MODEL_VERSION,
        "inputs": ["lookahead_x", "lookahead_y", "goal_distance"],
        "outputs": ["normalized_speed", "normalized_turn_rate"],
        "weights": {
            name: values.tolist()
            for name, values in network.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_network(path=DEFAULT_WEIGHTS_PATH):
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Neural follower model not found at {model_path}. "
            "Train it with `uv run --no-project --python /usr/bin/python3 "
            "train_neural_follower.py`."
        )
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if payload.get("version") != MODEL_VERSION:
        raise ValueError(
            f"Unsupported neural follower model version: {payload.get('version')!r}"
        )
    weights = {
        name: np.asarray(payload["weights"][name], dtype=float)
        for name in ("w1", "b1", "w2", "b2")
    }
    expected_shapes = {
        "w1": (INPUT_SIZE, HIDDEN_SIZE),
        "b1": (HIDDEN_SIZE,),
        "w2": (HIDDEN_SIZE, OUTPUT_SIZE),
        "b2": (OUTPUT_SIZE,),
    }
    for name, shape in expected_shapes.items():
        if weights[name].shape != shape or not np.isfinite(weights[name]).all():
            raise ValueError(f"Invalid {name} in neural follower model {model_path}")
    return weights


class NeuralFollower:
    """Small imitation-learned controller for study; it does not replace planning."""

    name = "Neural Imitation Follower"

    def __init__(
        self,
        occupancy_grid_map,
        robot_radius,
        max_speed=0.35,
        max_angular_speed=1.5,
        lookahead=0.55,
        clearance_margin=0.15,
        weights_path=DEFAULT_WEIGHTS_PATH,
    ):
        self.map = occupancy_grid_map
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.lookahead = lookahead
        self.clearance_margin = clearance_margin
        self.network = load_network(weights_path)

    def compute(self, pose, path, path_index, goal, dt):
        if not path or path_index >= len(path):
            return 0.0, 0.0

        target = path[-1]
        for point in path[path_index:]:
            if math.hypot(point[0] - pose[0], point[1] - pose[1]) >= self.lookahead:
                target = point
                break
        dx = target[0] - pose[0]
        dy = target[1] - pose[1]
        cosine = math.cos(pose[2])
        sine = math.sin(pose[2])
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        goal_distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        features = _features(
            local_x,
            local_y,
            goal_distance,
            self.lookahead,
        )
        normalized_speed, normalized_turn = predict_network(
            self.network,
            features,
        )
        velocity = max(0.0, min(1.0, float(normalized_speed))) * self.max_speed
        angular = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                float(normalized_turn) * self.max_angular_speed,
            ),
        )
        return velocity, angular
