"""Runs the UAV session lifetime experiments used in the paper."""

import os, json, math, time, warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
# Experiment settings
output_dir = "/content/drive/MyDrive/UAV_Paper_v4/"
os.makedirs(output_dir, exist_ok=True)
total_rounds = 30000
main_uav_count = 20
seed_count = 20
number_of_arms = 20
TAU_MIN = 10
TAU_MAX = 200
TAU_STEP = 10
TAU_BINS = np.arange(TAU_MIN, TAU_MAX + TAU_STEP, TAU_STEP)
UAV_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
scalability_rounds = 5000
X_MAX, Y_MAX, Z_MAX = (2000.0, 2000.0, 500.0)
V_MAX_UAV, V_MIN_UAV = (25.0, 5.0)
high_snr, low_snr = (1.0, 0.2)
high_to_low_probability, low_to_high_probability = (0.05, 0.1)
high_state_lifetime, low_state_lifetime = (120, 45)
completion_weight, drop_weight, exposure_weight, overhead_weight = (0.5, 0.4, 0.1, 0.05)
reward_noise = 0.05
hidden_units, regularization, exploration_scale = (64, 1.0, 0.5)
convergence_window = 500
convergence_threshold = 0.01
convergence_patience = 200
algorithms = [
    "Fixed-60s",
    "Fixed-120s",
    "Random",
    "Oracle",
    "UAAS-TS",
    "DS-TS",
    "SW-TS",
    "RewardTS",
    "LinUCB",
    "LinearTS",
    "NeuralTS",
]
plotted_algorithms = [
    "UAAS-TS",
    "DS-TS",
    "SW-TS",
    "RewardTS",
    "LinUCB",
    "LinearTS",
    "NeuralTS",
]
learning_algorithms = [
    "UAAS-TS",
    "DS-TS",
    "SW-TS",
    "RewardTS",
    "LinUCB",
    "LinearTS",
    "NeuralTS",
]
ENVIRONMENTS = ["Simple", "NonLinear", "MultiModal"]
plot_colors = {
    "Fixed-60s": "#AAAAAA",
    "Fixed-120s": "#888888",
    "Random": "#CCCCCC",
    "Oracle": "#4CAF50",
    "UAAS-TS": "#E63946",
    "DS-TS": "#FF8C00",
    "SW-TS": "#F4A261",
    "RewardTS": "#D4A017",
    "LinUCB": "#2A9D8F",
    "LinearTS": "#17BECF",
    "NeuralTS": "#264653",
}
line_styles = {
    "Fixed-60s": ":",
    "Fixed-120s": ":",
    "Random": ":",
    "Oracle": "-.",
    "UAAS-TS": "-",
    "DS-TS": "--",
    "SW-TS": "-.",
    "RewardTS": "--",
    "LinUCB": "-",
    "LinearTS": "--",
    "NeuralTS": "-",
}
MARKERS = {
    "Fixed-60s": "v",
    "Fixed-120s": "P",
    "Random": "x",
    "Oracle": "h",
    "UAAS-TS": "o",
    "DS-TS": "s",
    "SW-TS": "^",
    "RewardTS": "p",
    "LinUCB": "D",
    "LinearTS": "d",
    "NeuralTS": "*",
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 7,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# UAV movement and channel model
class UAV:
    """Keeps the position, speed and energy state of one UAV."""

    def __init__(self, uav_id, rng, uav_count):
        self.rng = rng
        self.uav_count = uav_count
        self.position = rng.uniform([0, 0, 50], [X_MAX, Y_MAX, Z_MAX])
        self.speed = rng.uniform(V_MIN_UAV, V_MAX_UAV)
        self.waypoint = rng.uniform([0, 0, 50], [X_MAX, Y_MAX, Z_MAX])
        self.energy = 1.0

    def step(self, round_limit=None):
        d = self.waypoint - self.position
        distance = np.linalg.norm(d)
        if distance < self.speed:
            self.position = self.waypoint.copy()
            self.waypoint = self.rng.uniform([0, 0, 50], [X_MAX, Y_MAX, Z_MAX])
            self.speed = self.rng.uniform(V_MIN_UAV, V_MAX_UAV)
        else:
            self.position += d / distance * self.speed
        remaining_path = self.waypoint - self.position
        heading = (
            np.arctan2(np.linalg.norm(remaining_path[:2]), remaining_path[2] + 1e-09)
            % np.pi
        )
        horizon = round_limit if round_limit else total_rounds
        self.energy = max(0.05, self.energy - 1.0 / (horizon * self.uav_count))
        return (self.position.copy(), self.energy, self.speed, heading)


def create_uavs(seed, uav_count):
    return [
        UAV(i, np.random.RandomState(seed * 1000 + i), uav_count)
        for i in range(uav_count)
    ]


def get_stable_lifetime(environment, channel, position, energy, v, heading):
    if environment == "Simple":
        return high_state_lifetime if channel == 1 else low_state_lifetime
    x, y, z = position
    snr = high_snr if channel == 1 else low_snr
    if environment == "NonLinear":
        jam = 1.0 if 800 <= x <= 1200 and 800 <= y <= 1200 else 0.0
        arg = (
            3.0 * snr
            - 0.1 * v
            - 0.005 * z
            - 0.5 * (heading / np.pi)
            + 2.0 * energy
            - 4.0 * jam
        )
        return TAU_MIN + (TAU_MAX - TAU_MIN) / (1.0 + np.exp(-arg))
    if environment == "MultiModal":
        jam1 = 1.0 if 400 <= x <= 800 and 400 <= y <= 800 else 0.0
        jam2 = 1.0 if 1200 <= x <= 1600 and 1200 <= y <= 1600 else 0.0
        jam3 = 1.0 if 800 <= x <= 1200 and 1400 <= y <= 1800 else 0.0
        spatial = np.sin(2 * np.pi * x / X_MAX) * np.cos(2 * np.pi * y / Y_MAX)
        los = 1.0 / (1.0 + np.exp(-0.05 * (z - 250)))
        normalized_speed = v / V_MAX_UAV
        doppler = -4.0 * (normalized_speed - 0.3) ** 2 + 0.5
        battery_term = 1.0 if energy >= 0.25 else -2.0
        heading_penalty = -2.0 * abs(np.sin(2 * heading))
        arg = (
            2.5 * snr
            + 1.5 * spatial
            + los
            + doppler
            + 1.5 * battery_term
            + heading_penalty
            - 3.0 * jam1
            - 3.5 * jam2
            - 2.5 * jam3
        )
        base = 1.0 / (1.0 + np.exp(-arg))
        modulation = (
            0.15 * np.sin(4 * np.pi * x / X_MAX) * np.cos(6 * np.pi * y / Y_MAX)
        )
        return TAU_MIN + (TAU_MAX - TAU_MIN) * np.clip(base + modulation, 0.05, 0.95)


def expected_reward(tau, stable_lifetime):
    drop_chance = (
        0.0
        if tau <= stable_lifetime
        else min(1.0, (tau - stable_lifetime) / (0.4 * TAU_MAX))
    )
    r = (
        completion_weight * (1 - drop_chance)
        - drop_weight * drop_chance
        - exposure_weight * (tau / TAU_MAX)
        - overhead_weight * (TAU_MAX / max(tau, 1))
    )
    return float(np.clip(r, -1, 1))


def sample_reward(arm, stable_lifetime):
    tau = TAU_BINS[arm]
    drop_chance = (
        0.0
        if tau <= stable_lifetime
        else min(1.0, (tau - stable_lifetime) / (0.4 * TAU_MAX))
    )
    drop = 1 if np.random.rand() < drop_chance else 0
    completed = 1 - drop
    r = (
        completion_weight * completed
        - drop_weight * drop
        - exposure_weight * (tau / TAU_MAX)
        - overhead_weight * (TAU_MAX / max(tau, 1))
    )
    r += reward_noise * np.random.randn()
    return (float(np.clip(r, -1, 1)), completed)


def build_context(channel, position, energy, v, heading):
    x, y, z = position
    snr = high_snr if channel == 1 else low_snr
    return np.array(
        [
            x / X_MAX,
            y / Y_MAX,
            z / Z_MAX,
            float(energy),
            snr,
            min(v / V_MAX_UAV, 1.0),
            heading / np.pi,
        ],
        dtype=np.float64,
    )


class TwoStateChannel:
    """Simple high and low channel model used in the experiment."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed + 42)
        self.state = 1

    def step(self):
        if self.state == 1 and self.rng.rand() < high_to_low_probability:
            self.state = 0
        elif self.state == 0 and self.rng.rand() < low_to_high_probability:
            self.state = 1
        return self.state


# Session lifetime methods
class FixedTimeout:
    """Always selects the same session timeout."""

    def __init__(self, number_of_arms, timeout_value):
        self.arm = int(np.argmin(np.abs(TAU_BINS - timeout_value)))

    def select(self, context=None):
        return self.arm

    def update(self, arm, r, completed, context=None):
        pass


class RandomTimeout:
    """Chooses a timeout at random."""

    def __init__(self, number_of_arms):
        self.arm_count = number_of_arms

    def select(self, context=None):
        return np.random.randint(self.arm_count)

    def update(self, arm, r, completed, context=None):
        pass


class OracleTimeout:
    """Chooses the arm with the highest expected reward."""

    def __init__(self, number_of_arms):
        self.arm_count = number_of_arms
        self.best_arm = 0

    def set_oracle(self, stable_lifetime):
        self.best_arm = int(
            np.argmax(
                [
                    expected_reward(TAU_BINS[k], stable_lifetime)
                    for k in range(self.arm_count)
                ]
            )
        )

    def select(self, context=None):
        return self.best_arm

    def update(self, arm, r, completed, context=None):
        pass


class UAAS_TS:
    """Thompson Sampling with binary session completion feedback."""

    def __init__(self, number_of_arms):
        self.a = np.ones(number_of_arms)
        self.b = np.ones(number_of_arms)

    def select(self, context=None):
        return int(np.argmax(np.random.beta(self.a, self.b)))

    def update(self, arm, r, completed, context=None):
        self.a[arm] += completed
        self.b[arm] += 1 - completed


class DiscountedTS:
    """Thompson Sampling with discounted observations."""

    def __init__(self, number_of_arms, gamma=0.99):
        self.discount = gamma
        self.a = np.ones(number_of_arms)
        self.b = np.ones(number_of_arms)

    def select(self, context=None):
        return int(
            np.argmax(
                np.random.beta(np.maximum(self.a, 1e-06), np.maximum(self.b, 1e-06))
            )
        )

    def update(self, arm, r, completed, context=None):
        self.a *= self.discount
        self.b *= self.discount
        self.a[arm] += completed
        self.b[arm] += 1 - completed


class SlidingWindowTS:
    """Thompson Sampling using a fixed recent window."""

    def __init__(self, number_of_arms, W=200):
        self.arm_count = number_of_arms
        self.window_size = W
        self.history = deque()

    def select(self, context=None):
        a, b = (np.ones(self.arm_count), np.ones(self.arm_count))
        for arm, completed in self.history:
            a[arm] += completed
            b[arm] += 1 - completed
        return int(np.argmax(np.random.beta(a, b)))

    def update(self, arm, r, completed, context=None):
        self.history.append((arm, completed))
        if len(self.history) > self.window_size:
            self.history.popleft()


class RewardTS:
    """Context-free Thompson Sampling using the scalar reward."""

    def __init__(self, number_of_arms):
        self.arm_count = number_of_arms
        self.prior_mean = 0.0
        self.prior_strength = 1.0
        self.prior_shape = 2.0
        self.prior_scale = 0.5
        self.n = np.zeros(number_of_arms)
        self.reward_sum = np.zeros(number_of_arms)
        self.squared_reward_sum = np.zeros(number_of_arms)

    def select(self, context=None):
        samples = np.zeros(self.arm_count)
        for k in range(self.arm_count):
            n = self.n[k]
            kappa_n = self.prior_strength + n
            mu_n = (
                self.prior_strength * self.prior_mean + self.reward_sum[k]
            ) / kappa_n
            alpha_n = self.prior_shape + n / 2.0
            mean_r = self.reward_sum[k] / (n + 1e-09)
            beta_n = (
                self.prior_scale
                + 0.5
                * (self.squared_reward_sum[k] - self.reward_sum[k] ** 2 / (n + 1e-09))
                + 0.5
                * self.prior_strength
                * n
                * (mean_r - self.prior_mean) ** 2
                / kappa_n
            )
            beta_n = max(beta_n, 1e-06)
            var = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
            samples[k] = np.random.normal(mu_n, np.sqrt(var / kappa_n))
        return int(np.argmax(samples))

    def update(self, arm, r, completed, context=None):
        self.n[arm] += 1
        self.reward_sum[arm] += r
        self.squared_reward_sum[arm] += r * r


class LinUCB:
    """Disjoint LinUCB with a Sherman-Morrison matrix update."""

    def __init__(self, number_of_arms, d=7, alpha=1.0):
        self.arm_count = number_of_arms
        self.alpha = alpha
        self.inverse_matrices = [np.eye(d) for _ in range(number_of_arms)]
        self.b = [np.zeros(d) for _ in range(number_of_arms)]

    def select(self, context):
        ucb = np.zeros(self.arm_count)
        for k in range(self.arm_count):
            theta = self.inverse_matrices[k] @ self.b[k]
            ucb[k] = theta @ context + self.alpha * np.sqrt(
                context @ self.inverse_matrices[k] @ context
            )
        return int(np.argmax(ucb))

    def update(self, arm, r, completed, context):
        matrix_context = self.inverse_matrices[arm] @ context
        self.inverse_matrices[arm] -= np.outer(matrix_context, matrix_context) / (
            1.0 + float(context @ matrix_context)
        )
        self.b[arm] += r * context


class LinearTS:
    """Linear contextual Thompson Sampling."""

    def __init__(self, number_of_arms, d=7, lam=1.0, nu=0.5):
        self.arm_count = number_of_arms
        self.exploration = nu
        self.d = d
        self.inverse_matrices = [np.eye(d) / lam for _ in range(number_of_arms)]
        self.b = [np.zeros(d) for _ in range(number_of_arms)]

    def select(self, context):
        samples = np.zeros(self.arm_count)
        for k in range(self.arm_count):
            mu = self.inverse_matrices[k] @ self.b[k]
            var = self.exploration * (context @ self.inverse_matrices[k] @ context)
            var = max(var, 1e-09)
            samples[k] = np.random.normal(float(mu @ context), np.sqrt(var))
        return int(np.argmax(samples))

    def update(self, arm, r, completed, context):
        matrix_context = self.inverse_matrices[arm] @ context
        self.inverse_matrices[arm] -= np.outer(matrix_context, matrix_context) / (
            1.0 + float(context @ matrix_context)
        )
        self.b[arm] += r * context


class NeuralTS:
    """Neural Thompson Sampling with a two-layer feature network."""

    RETRAIN_EVERY = 50
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64

    def __init__(
        self,
        number_of_arms,
        d=7,
        h=hidden_units,
        lam=regularization,
        nu=exploration_scale,
    ):
        self.arm_count = number_of_arms
        self.hidden_size = h
        self.exploration = nu
        self.update_count = 0
        s = 1.0 / np.sqrt(h)
        self.first_layer_weights = np.random.randn(h, d) * s
        self.first_layer_bias = np.zeros(h)
        self.second_layer_weights = np.random.randn(h, h) * s
        self.second_layer_bias = np.zeros(h)
        self.H_inv = [1.0 / lam * np.eye(h) for _ in range(number_of_arms)]
        self.response_vectors = [np.zeros(h) for _ in range(number_of_arms)]
        self.replay_buffer = []

    def _fwd(self, x):
        first_linear = self.first_layer_weights @ x + self.first_layer_bias
        first_hidden = np.maximum(0, first_linear)
        second_linear = (
            self.second_layer_weights @ first_hidden + self.second_layer_bias
        )
        second_hidden = np.maximum(0, second_linear)
        return (first_linear, first_hidden, second_linear, second_hidden)

    def _feat(self, x):
        return self._fwd(x)[3]

    def _train(self, arm):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return
        batch = self.replay_buffer[-self.BATCH_SIZE :]
        first_weight_gradient = np.zeros_like(self.first_layer_weights)
        first_bias_gradient = np.zeros_like(self.first_layer_bias)
        second_weight_gradient = np.zeros_like(self.second_layer_weights)
        second_bias_gradient = np.zeros_like(self.second_layer_bias)
        for stored_context, a, r in batch:
            theta = self.H_inv[a] @ self.response_vectors[a]
            first_linear, first_hidden, second_linear, second_hidden = self._fwd(
                stored_context
            )
            error = float(second_hidden @ theta) - r
            second_gradient = error * theta
            second_relu_gradient = second_gradient * (second_linear > 0)
            second_weight_gradient += np.outer(second_relu_gradient, first_hidden)
            second_bias_gradient += second_relu_gradient
            first_gradient = self.second_layer_weights.T @ second_relu_gradient
            first_relu_gradient = first_gradient * (first_linear > 0)
            first_weight_gradient += np.outer(first_relu_gradient, stored_context)
            first_bias_gradient += first_relu_gradient
        n = len(batch)
        self.first_layer_weights -= self.LEARNING_RATE * first_weight_gradient / n
        self.first_layer_bias -= self.LEARNING_RATE * first_bias_gradient / n
        self.second_layer_weights -= self.LEARNING_RATE * second_weight_gradient / n
        self.second_layer_bias -= self.LEARNING_RATE * second_bias_gradient / n

    def select(self, context):
        features = self._feat(context)
        samples = np.zeros(self.arm_count)
        for k in range(self.arm_count):
            theta = self.H_inv[k] @ self.response_vectors[k]
            mu = float(features @ theta)
            var = max(
                1e-09,
                float(
                    self.exploration
                    / np.sqrt(self.hidden_size)
                    * (features @ self.H_inv[k] @ features)
                ),
            )
            samples[k] = np.random.normal(mu, np.sqrt(var))
        return int(np.argmax(samples))

    def update(self, arm, r, completed, context):
        self.replay_buffer.append((context.copy(), arm, r))
        features = self._feat(context)
        matrix_features = self.H_inv[arm] @ features
        self.H_inv[arm] -= np.outer(matrix_features, matrix_features) / (
            1.0 + float(features @ matrix_features)
        )
        self.response_vectors[arm] += r * features
        self.update_count += 1
        if self.update_count % self.RETRAIN_EVERY == 0:
            self._train(arm)


def create_methods():
    return {
        "Fixed-60s": FixedTimeout(number_of_arms, 60),
        "Fixed-120s": FixedTimeout(number_of_arms, 120),
        "Random": RandomTimeout(number_of_arms),
        "Oracle": OracleTimeout(number_of_arms),
        "UAAS-TS": UAAS_TS(number_of_arms),
        "DS-TS": DiscountedTS(number_of_arms),
        "SW-TS": SlidingWindowTS(number_of_arms),
        "RewardTS": RewardTS(number_of_arms),
        "LinUCB": LinUCB(number_of_arms),
        "LinearTS": LinearTS(number_of_arms),
        "NeuralTS": NeuralTS(number_of_arms),
    }


# Main simulation
def run_single_seed(seed, environment, uav_count, round_count=total_rounds):
    """Runs one seed and returns the per-round results."""
    np.random.seed(seed)
    channel = TwoStateChannel(seed)
    uavs = create_uavs(seed, uav_count)
    methods = create_methods()
    cumulative_regret = {a: np.zeros(round_count) for a in algorithms}
    reward_history = {a: np.zeros(round_count) for a in algorithms}
    completion_history = {a: np.zeros(round_count) for a in algorithms}
    instant_regret = {a: np.zeros(round_count) for a in algorithms}
    running_regret = {a: 0.0 for a in algorithms}
    for t in range(round_count):
        channel_state = channel.step()
        uav = uavs[t % uav_count]
        position, energy, v, heading = uav.step(round_count)
        context = build_context(channel_state, position, energy, v, heading)
        stable_lifetime = get_stable_lifetime(
            environment, channel_state, position, energy, v, heading
        )
        arm_rewards = [expected_reward(tau, stable_lifetime) for tau in TAU_BINS]
        best_expected_reward = max(arm_rewards)
        methods["Oracle"].set_oracle(stable_lifetime)
        for name in algorithms:
            arm = methods[name].select(context)
            observed_reward, completed = sample_reward(arm, stable_lifetime)
            round_regret = max(
                0.0,
                best_expected_reward - expected_reward(TAU_BINS[arm], stable_lifetime),
            )
            methods[name].update(arm, observed_reward, completed, context)
            running_regret[name] += round_regret
            cumulative_regret[name][t] = running_regret[name]
            reward_history[name][t] = observed_reward
            completion_history[name][t] = completed
            instant_regret[name][t] = round_regret
        if (t + 1) % 5000 == 0:
            print(
                f"    Seed {seed + 1:02d} | {t + 1}/{round_count} NTS={running_regret['NeuralTS']:.0f} LTS={running_regret['LinearTS']:.0f} Lin={running_regret['LinUCB']:.0f} RTS={running_regret['RewardTS']:.0f}",
                end="\r",
            )
    return (cumulative_regret, reward_history, completion_history, instant_regret)


def run_environment(
    environment,
    uav_count=main_uav_count,
    number_of_seeds=seed_count,
    round_count=total_rounds,
):
    """Runs all requested seeds for one channel environment."""
    print(
        f"\n  Running {environment} | N={uav_count} | {number_of_seeds} seeds | T={round_count} ..."
    )
    all_cumulative_regret = {
        a: np.zeros((number_of_seeds, round_count)) for a in algorithms
    }
    all_rewards = {a: np.zeros((number_of_seeds, round_count)) for a in algorithms}
    all_completions = {a: np.zeros((number_of_seeds, round_count)) for a in algorithms}
    all_instant_regret = {
        a: np.zeros((number_of_seeds, round_count)) for a in algorithms
    }
    for s in range(number_of_seeds):
        cumulative_result, reward_result, completion_result, instant_result = (
            run_single_seed(s, environment, uav_count, round_count)
        )
        for a in algorithms:
            all_cumulative_regret[a][s] = cumulative_result[a]
            all_rewards[a][s] = reward_result[a]
            all_completions[a][s] = completion_result[a]
            all_instant_regret[a][s] = instant_result[a]
    print(
        f"\n    {environment} complete.                                              "
    )
    return (all_cumulative_regret, all_rewards, all_completions, all_instant_regret)


def find_convergence_rounds(
    all_instant_regret,
    algorithm,
    window=convergence_window,
    threshold=convergence_threshold,
    patience=convergence_patience,
):
    """Finds the convergence round for each seed."""
    number_of_seeds = all_instant_regret[algorithm].shape[0]
    conv_rounds = []
    for s in range(number_of_seeds):
        instant_result = all_instant_regret[algorithm][s]
        rolling = np.convolve(instant_result, np.ones(window) / window, mode="valid")
        below = 0
        found = None
        for i, val in enumerate(rolling):
            if val < threshold:
                below += 1
                if below >= patience:
                    found = i + window - patience
                    break
            else:
                below = 0
        conv_rounds.append(found)
    return conv_rounds


def calculate_metrics(
    all_cumulative_regret, all_rewards, all_completions, all_instant_regret
):
    metrics = {}
    for a in algorithms:
        final_regrets = all_cumulative_regret[a][:, -1]
        if a in learning_algorithms:
            convergence_values = find_convergence_rounds(all_instant_regret, a)
            finished_values = [c for c in convergence_values if c is not None]
            convergence_count = len(finished_values)
            mean_convergence_round = (
                float(np.mean(finished_values)) if finished_values else float("inf")
            )
        else:
            convergence_count = "N/A"
            mean_convergence_round = float("nan")
        metrics[a] = {
            "regret_mean": float(np.mean(final_regrets)),
            "regret_std": float(np.std(final_regrets)),
            "regret_ci95": float(1.96 * np.std(final_regrets) / np.sqrt(seed_count)),
            "avg_reward": float(np.mean(all_rewards[a])),
            "completion": float(np.mean(all_completions[a]) * 100),
            "steady_compl": float(np.mean(all_completions[a][:, -500:]) * 100),
            "conv_round": mean_convergence_round,
            "conv_count": convergence_count,
            "final_regrets": final_regrets.tolist(),
        }
    uaas_regret = metrics["UAAS-TS"]["regret_mean"]
    for a in algorithms:
        metrics[a]["delta_uaas"] = (
            (uaas_regret - metrics[a]["regret_mean"]) / uaas_regret * 100
            if uaas_regret > 0
            else 0
        )
    fixed_algos = ["Fixed-60s", "Fixed-120s"]
    best_fixed = min(fixed_algos, key=lambda x: metrics[x]["regret_mean"])
    best_fixed_regret = metrics[best_fixed]["regret_mean"]
    for a in algorithms:
        metrics[a]["delta_best_fixed"] = (
            (best_fixed_regret - metrics[a]["regret_mean"]) / best_fixed_regret * 100
            if best_fixed_regret > 0
            else 0
        )
    metrics["_best_fixed"] = best_fixed
    return metrics


def calculate_significance(metrics):
    """Runs paired tests for the main method comparisons."""
    pairs = [
        ("LinUCB", "NeuralTS"),
        ("LinearTS", "NeuralTS"),
        ("LinUCB", "LinearTS"),
        ("RewardTS", "LinUCB"),
        ("RewardTS", "NeuralTS"),
        ("UAAS-TS", "RewardTS"),
    ]
    results = {}
    for a1, a2 in pairs:
        r1 = np.array(metrics[a1]["final_regrets"])
        r2 = np.array(metrics[a2]["final_regrets"])
        t_stat, t_pval = scipy_stats.ttest_rel(r1, r2)
        try:
            w_stat, w_pval = scipy_stats.wilcoxon(r1, r2, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = (float("nan"), float("nan"))
        results[f"{a1}_vs_{a2}"] = {
            "mean_diff": float(np.mean(r1 - r2)),
            "t_stat": float(t_stat),
            "t_pval": float(t_pval),
            "w_stat": float(w_stat),
            "w_pval": float(w_pval),
            "sig_005": bool(t_pval < 0.05),
            "sig_001": bool(t_pval < 0.01),
        }
    return results


def print_results(experiment_metrics, all_significance_tests):
    for env in ENVIRONMENTS:
        m = experiment_metrics[env]
        print(f"\n{'=' * 115}")
        print(f"  {env.upper()} CHANNEL | Best fixed: {m['_best_fixed']}")
        print(f"{'=' * 115}")
        print(
            f"  {'Algorithm':<12} {'Regret':>10} {'±CI95':>8} {'Conv':>8} {'#C':>4} {'AvgRew':>9} {'Compl%':>8} {'Δ%vsTS':>8} {'Δ%vsBF':>8}"
        )
        print(f"  {'-' * 105}")
        for a in algorithms:
            d = m[a]
            if a in learning_algorithms:
                conv_str = (
                    f"{d['conv_round']:.0f}"
                    if d["conv_round"] < float("inf")
                    else ">MAX"
                )
                cc = f"{d['conv_count']}/{seed_count}"
            else:
                conv_str = "—"
                cc = "—"
            print(
                f"  {a:<12} {d['regret_mean']:>10.1f} {d['regret_ci95']:>8.1f} {conv_str:>8} {cc:>4} {d['avg_reward']:>9.4f} {d['completion']:>8.1f} {d['delta_uaas']:>7.1f}% {d['delta_best_fixed']:>7.1f}%"
            )
        significance_results = all_significance_tests[env]
        print(f"\n  Significance tests (paired, 20 seeds):")
        print(
            f"  {'Pair':<25} {'MeanDiff':>10} {'t-pval':>10} {'W-pval':>10} {'Sig':>8}"
        )
        print(f"  {'-' * 68}")
        for key, s in significance_results.items():
            tag = "***" if s["sig_001"] else "*" if s["sig_005"] else "ns"
            print(
                f"  {key:<25} {s['mean_diff']:>10.1f} {s['t_pval']:>10.4f} {s['w_pval']:>10.4f} {tag:>8}"
            )


# Figures
def plot_regret_curves(
    all_cumulative_regret,
    env_name,
    filename,
    algos=plotted_algorithms,
    round_count=total_rounds,
):
    """Regret curves with CI bands + zoomed contextual panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    rounds = np.arange(1, round_count + 1)
    for a in algos:
        mean = np.mean(all_cumulative_regret[a], axis=0)
        ci = 1.96 * np.std(all_cumulative_regret[a], axis=0) / np.sqrt(seed_count)
        ax1.plot(
            rounds,
            mean,
            label=a,
            color=plot_colors[a],
            linestyle=line_styles[a],
            linewidth=1.5,
        )
        ax1.fill_between(rounds, mean - ci, mean + ci, color=plot_colors[a], alpha=0.1)
    ax1.set_xlabel("Round ($t$)")
    ax1.set_ylabel("Cumulative Regret")
    ax1.set_title(f"{env_name} — All Learning Algorithms")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2, ls="--")
    for a in ["LinUCB", "LinearTS", "NeuralTS"]:
        mean = np.mean(all_cumulative_regret[a], axis=0)
        ci = 1.96 * np.std(all_cumulative_regret[a], axis=0) / np.sqrt(seed_count)
        ax2.plot(rounds, mean, label=a, color=plot_colors[a], linewidth=2)
        ax2.fill_between(rounds, mean - ci, mean + ci, color=plot_colors[a], alpha=0.15)
    ax2.set_xlabel("Round ($t$)")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title(f"{env_name} — Contextual (Zoomed)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_reward_curves(
    all_rewards, env_name, filename, window=200, round_count=total_rounds
):
    """Average reward curves."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in plotted_algorithms:
        mr = np.mean(all_rewards[a], axis=0)
        sm = np.convolve(mr, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, window + len(sm)),
            sm,
            label=a,
            color=plot_colors[a],
            linestyle=line_styles[a],
            linewidth=1.5,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Avg Reward (w={window})")
    ax.set_title(f"Average Reward — {env_name}")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_completion_curves(
    all_completions, env_name, filename, window=300, round_count=total_rounds
):
    """Session completion rate over time."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in plotted_algorithms:
        mc = np.mean(all_completions[a], axis=0)
        sm = np.convolve(mc, np.ones(window) / window, mode="valid") * 100
        ax.plot(
            np.arange(window, window + len(sm)),
            sm,
            label=a,
            color=plot_colors[a],
            linestyle=line_styles[a],
            linewidth=1.5,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title(f"Session Completion — {env_name}")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.2, ls="--")
    ax.set_ylim([0, 105])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_instant_regret(
    all_instant_regret, env_name, filename, window=500, round_count=total_rounds
):
    """Instantaneous regret (smoothed) showing convergence."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in plotted_algorithms:
        mi = np.mean(all_instant_regret[a], axis=0)
        sm = np.convolve(mi, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, window + len(sm)),
            sm,
            label=a,
            color=plot_colors[a],
            linestyle=line_styles[a],
            linewidth=1.5,
        )
    ax.axhline(
        y=convergence_threshold,
        color="gray",
        ls=":",
        alpha=0.6,
        label="Conv. threshold",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Per-Round Regret (smoothed)")
    ax.set_title(f"Instantaneous Regret — {env_name}")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_all_methods(experiment_metrics, filename):
    """Bar chart: final regret across all 3 environments (learning algos)."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    bar_algos = learning_algorithms
    x = np.arange(len(bar_algos))
    w = 0.25
    colors_env = ["#2A9D8F", "#E63946", "#7B2D8E"]
    for i, env in enumerate(ENVIRONMENTS):
        vals = [experiment_metrics[env][a]["regret_mean"] for a in bar_algos]
        ci = [experiment_metrics[env][a]["regret_ci95"] for a in bar_algos]
        ax.bar(
            x + i * w,
            vals,
            w,
            yerr=ci,
            label=env,
            color=colors_env[i],
            alpha=0.85,
            capsize=3,
            edgecolor="white",
        )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"Final Regret — All Environments (T={total_rounds})")
    ax.set_xticks(x + w)
    ax.set_xticklabels(bar_algos, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_contextual_methods(experiment_metrics, filename):
    """Bar chart: contextual methods only (LinUCB vs LinearTS vs NeuralTS)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ctx_algos = ["LinUCB", "LinearTS", "NeuralTS"]
    x = np.arange(len(ctx_algos))
    w = 0.25
    colors_env = ["#2A9D8F", "#E63946", "#7B2D8E"]
    for i, env in enumerate(ENVIRONMENTS):
        vals = [experiment_metrics[env][a]["regret_mean"] for a in ctx_algos]
        ci = [experiment_metrics[env][a]["regret_ci95"] for a in ctx_algos]
        bars = ax.bar(
            x + i * w,
            vals,
            w,
            yerr=ci,
            label=env,
            color=colors_env[i],
            alpha=0.85,
            capsize=4,
            edgecolor="white",
        )
        for j, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci[j] + 10,
                f"{vals[j]:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Contextual Methods — All Environments")
    ax.set_xticks(x + w)
    ax.set_xticklabels(ctx_algos)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_convergence(experiment_metrics, filename):
    """Convergence round bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_algos = learning_algorithms
    x = np.arange(len(bar_algos))
    w = 0.25
    colors_env = ["#2A9D8F", "#E63946", "#7B2D8E"]
    for i, env in enumerate(ENVIRONMENTS):
        vals = [
            min(experiment_metrics[env][a]["conv_round"], total_rounds)
            for a in bar_algos
        ]
        ax.bar(
            x + i * w,
            vals,
            w,
            label=env,
            color=colors_env[i],
            alpha=0.85,
            edgecolor="white",
        )
        for j, v in enumerate(vals):
            lbl = f"{v:.0f}" if v < total_rounds else ">MAX"
            ax.text(x[j] + i * w, v + 200, lbl, ha="center", fontsize=5.5, rotation=45)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Convergence Round")
    ax.set_title("Convergence Round — All Environments")
    ax.set_xticks(x + w)
    ax.set_xticklabels(bar_algos, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


def plot_scalability(scale_data, filename):
    """Regret vs N_UAV for both environments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, env, title in zip(
        axes, ["Simple", "MultiModal"], ["Simple Channel", "Multi-Modal Channel"]
    ):
        for algorithm in learning_algorithms:
            if algorithm in scale_data[UAV_COUNTS[0]][env]:
                means = [scale_data[n][env][algorithm]["mean"] for n in UAV_COUNTS]
                ci = [scale_data[n][env][algorithm]["ci95"] for n in UAV_COUNTS]
                ax.errorbar(
                    UAV_COUNTS,
                    means,
                    yerr=ci,
                    label=algorithm,
                    color=plot_colors[algorithm],
                    marker=MARKERS[algorithm],
                    linewidth=1.5,
                    markersize=5,
                    capsize=3,
                )
        ax.set_xlabel("Number of UAVs")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(title)
        ax.set_xticks(UAV_COUNTS)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"  [✓] {filename}")


# Timing and scalability checks
def run_timing_test():
    print("\n  Timing benchmark (10k iters, 500 warmup) ...")
    context = np.random.rand(7)
    n_trials = 10000
    n_warm = 500
    results = {}
    bench_agents = [
        ("UAAS-TS", UAAS_TS(number_of_arms)),
        ("DS-TS", DiscountedTS(number_of_arms)),
        ("SW-TS", SlidingWindowTS(number_of_arms)),
        ("RewardTS", RewardTS(number_of_arms)),
        ("LinUCB", LinUCB(number_of_arms)),
        ("LinearTS", LinearTS(number_of_arms)),
        ("NeuralTS", NeuralTS(number_of_arms)),
    ]
    for name, agent in bench_agents:
        for _ in range(n_warm):
            arm = agent.select(context)
            agent.update(arm, 0.1, 1, context)
        sel_t, upd_t = ([], [])
        for _ in range(n_trials):
            t0 = time.perf_counter()
            arm = agent.select(context)
            t1 = time.perf_counter()
            agent.update(arm, 0.1, 1, context)
            t2 = time.perf_counter()
            sel_t.append((t1 - t0) * 1000000.0)
            upd_t.append((t2 - t1) * 1000000.0)
        sel_t = np.array(sel_t)
        upd_t = np.array(upd_t)
        total = sel_t + upd_t
        results[name] = {
            "select_mean": float(np.mean(sel_t)),
            "select_std": float(np.std(sel_t)),
            "update_mean": float(np.mean(upd_t)),
            "update_std": float(np.std(upd_t)),
            "total_mean": float(np.mean(total)),
            "total_median": float(np.median(total)),
            "total_p95": float(np.percentile(total, 95)),
            "total_p99": float(np.percentile(total, 99)),
            "total_max": float(np.max(total)),
        }
        print(
            f"    {name:<12} mean={results[name]['total_mean']:>7.1f}  p50={results[name]['total_median']:>7.1f}  p95={results[name]['total_p95']:>7.1f}  p99={results[name]['total_p99']:>7.1f}  max={results[name]['total_max']:>9.1f} μs"
        )
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [✓] Timing saved")
    return results


def run_scalability_test():
    print(f"\n{'=' * 60}\n  SCALABILITY SWEEP (T={scalability_rounds})\n{'=' * 60}")
    scale_data = {}
    total = len(UAV_COUNTS) * 2
    count = 0
    for uav_count in UAV_COUNTS:
        scale_data[uav_count] = {}
        for env in ["Simple", "MultiModal"]:
            count += 1
            print(f"  [{count}/{total}] N={uav_count} | {env}")
            finals = {a: np.zeros(seed_count) for a in learning_algorithms}
            for seed in range(seed_count):
                np.random.seed(seed)
                channel = TwoStateChannel(seed)
                uavs = create_uavs(seed, uav_count)
                agents_s = {
                    "UAAS-TS": UAAS_TS(number_of_arms),
                    "DS-TS": DiscountedTS(number_of_arms),
                    "SW-TS": SlidingWindowTS(number_of_arms),
                    "RewardTS": RewardTS(number_of_arms),
                    "LinUCB": LinUCB(number_of_arms),
                    "LinearTS": LinearTS(number_of_arms),
                    "NeuralTS": NeuralTS(number_of_arms),
                }
                cum = {a: 0.0 for a in learning_algorithms}
                for t in range(scalability_rounds):
                    channel_state = channel.step()
                    uav = uavs[t % uav_count]
                    position, energy, v, heading = uav.step(scalability_rounds)
                    context = build_context(channel_state, position, energy, v, heading)
                    stable_lifetime = get_stable_lifetime(
                        env, channel_state, position, energy, v, heading
                    )
                    arm_rewards = [
                        expected_reward(tau, stable_lifetime) for tau in TAU_BINS
                    ]
                    best_expected_reward = max(arm_rewards)
                    for name in learning_algorithms:
                        arm = agents_s[name].select(context)
                        observed_reward, completed = sample_reward(arm, stable_lifetime)
                        round_regret = max(
                            0.0,
                            best_expected_reward
                            - expected_reward(TAU_BINS[arm], stable_lifetime),
                        )
                        agents_s[name].update(arm, observed_reward, completed, context)
                        cum[name] += round_regret
                for a in learning_algorithms:
                    finals[a][seed] = cum[a]
            summary = {}
            for a in learning_algorithms:
                summary[a] = {
                    "mean": round(float(np.mean(finals[a])), 2),
                    "std": round(float(np.std(finals[a])), 2),
                    "ci95": round(
                        float(1.96 * np.std(finals[a]) / np.sqrt(seed_count)), 2
                    ),
                }
            scale_data[uav_count][env] = summary
            print(
                f"    NTS={summary['NeuralTS']['mean']:.0f} LTS={summary['LinearTS']['mean']:.0f} Lin={summary['LinUCB']['mean']:.0f}"
            )
    with open(os.path.join(output_dir, "scalability.json"), "w") as f:
        json.dump(scale_data, f, indent=2)
    return scale_data


if __name__ == "__main__":
    t_start = time.time()
    print("\n" + "*" * 65)
    print("  UAV session lifetime experiment")
    print(f"  Algos: {len(algorithms)} | Envs: {ENVIRONMENTS}")
    print(f"  N={main_uav_count} | Seeds={seed_count} | T={total_rounds}")
    print("*" * 65)
    experiment_metrics = {}
    all_significance_tests = {}
    experiment_data = {}
    for env in ENVIRONMENTS:
        print(f"\n{'=' * 60}\n  ENVIRONMENT: {env}\n{'=' * 60}")
        cumulative_result, reward_result, completion_result, instant_result = (
            run_environment(env)
        )
        experiment_data[env] = (
            cumulative_result,
            reward_result,
            completion_result,
            instant_result,
        )
        save_dict = {}
        for a in algorithms:
            save_dict[f"{a}_regret"] = cumulative_result[a]
            save_dict[f"{a}_reward"] = reward_result[a]
        np.savez_compressed(
            os.path.join(output_dir, f"raw_{env.lower()}.npz"), **save_dict
        )
        metrics = calculate_metrics(
            cumulative_result, reward_result, completion_result, instant_result
        )
        experiment_metrics[env] = metrics
        significance_results = calculate_significance(metrics)
        all_significance_tests[env] = significance_results
    print(f"\n{'=' * 60}\n  COMPLETE RESULTS\n{'=' * 60}")
    print_results(experiment_metrics, all_significance_tests)
    save_m = {}
    for env in ENVIRONMENTS:
        save_m[env] = {}
        for a in algorithms:
            d = dict(experiment_metrics[env][a])
            d.pop("final_regrets", None)
            save_m[env][a] = d
        save_m[env]["_best_fixed"] = experiment_metrics[env]["_best_fixed"]
    with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
        json.dump(save_m, f, indent=2)
    with open(os.path.join(output_dir, "significance_tests.json"), "w") as f:
        json.dump(all_significance_tests, f, indent=2)
    print(f"\n{'=' * 60}\n  GENERATING FIGURES\n{'=' * 60}")
    env_titles = {
        "Simple": "Simple Channel (Env A)",
        "NonLinear": "Non-Linear Channel (Env B)",
        "MultiModal": "Multi-Modal Channel (Env C)",
    }
    for env in ENVIRONMENTS:
        cumulative_result, reward_result, completion_result, instant_result = (
            experiment_data[env]
        )
        tag = env.lower()
        plot_regret_curves(cumulative_result, env_titles[env], f"fig_regret_{tag}.pdf")
        plot_reward_curves(reward_result, env_titles[env], f"fig_reward_{tag}.pdf")
        plot_completion_curves(
            completion_result, env_titles[env], f"fig_completion_{tag}.pdf"
        )
        plot_instant_regret(
            instant_result, env_titles[env], f"fig_inst_regret_{tag}.pdf"
        )
    plot_all_methods(experiment_metrics, "fig_bar_all_envs.pdf")
    plot_contextual_methods(experiment_metrics, "fig_contextual_zoom.pdf")
    plot_convergence(experiment_metrics, "fig_convergence_bars.pdf")
    print(f"\n{'=' * 60}\n  TIMING BENCHMARK\n{'=' * 60}")
    timing = run_timing_test()
    scale_data = run_scalability_test()
    plot_scalability(scale_data, "fig_scalability.pdf")
    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 60}")
    print(f"  ALL COMPLETE — {total_min:.1f} minutes")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")
    print("\n  Files:")
    for f in sorted(os.listdir(output_dir)):
        sz = os.path.getsize(os.path.join(output_dir, f))
        print(f"    {f:50s} {sz / 1024:>8.1f} KB")
