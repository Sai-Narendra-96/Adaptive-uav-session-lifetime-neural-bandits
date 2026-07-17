"""Creates one CSV file for each simulation seed in Google Colab."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np

# Settings
use_google_drive = True
drive_output_folder = "/content/drive/MyDrive/UAV_Paper_v4/UAV Simulation Datasets"
rounds_per_seed = 30000
first_seed = 0
number_of_seeds = 20
number_of_uavs = 20
selected_environment = "all"
save_all_arm_rewards = False
progress_interval = 5000
# Values used in the experiment
TAU_MIN = 10
TAU_MAX = 200
TAU_STEP = 10
TAU_BINS = np.arange(TAU_MIN, TAU_MAX + TAU_STEP, TAU_STEP, dtype=np.int64)
X_MAX, Y_MAX, Z_MAX = (2000.0, 2000.0, 500.0)
V_MAX_UAV, V_MIN_UAV = (25.0, 5.0)
high_snr, low_snr = (1.0, 0.2)
high_to_low_probability, low_to_high_probability = (0.05, 0.1)
high_state_lifetime, low_state_lifetime = (120.0, 45.0)
completion_weight, drop_weight, exposure_weight, overhead_weight = (0.5, 0.4, 0.1, 0.05)
ENVIRONMENTS = ("Simple", "NonLinear", "MultiModal")


# UAV movement and channel model
class UAV:
    """Stores the movement and energy state of one UAV."""

    def __init__(self, uav_id: int, rng: np.random.RandomState, uav_count: int) -> None:
        self.uid = uav_id
        self.rng = rng
        self.uav_count = uav_count
        self.position = rng.uniform([0.0, 0.0, 50.0], [X_MAX, Y_MAX, Z_MAX])
        self.speed = float(rng.uniform(V_MIN_UAV, V_MAX_UAV))
        self.waypoint = rng.uniform([0.0, 0.0, 50.0], [X_MAX, Y_MAX, Z_MAX])
        self.energy = 1.0

    def step(self, total_rounds: int) -> tuple[np.ndarray, float, float, float]:
        direction = self.waypoint - self.position
        distance = float(np.linalg.norm(direction))
        if distance < self.speed:
            self.position = self.waypoint.copy()
            self.waypoint = self.rng.uniform([0.0, 0.0, 50.0], [X_MAX, Y_MAX, Z_MAX])
            self.speed = float(self.rng.uniform(V_MIN_UAV, V_MAX_UAV))
        else:
            self.position += direction / distance * self.speed
        remaining = self.waypoint - self.position
        heading = float(
            np.arctan2(np.linalg.norm(remaining[:2]), remaining[2] + 1e-09) % np.pi
        )
        self.energy = max(0.05, self.energy - 1.0 / (total_rounds * self.uav_count))
        return (self.position.copy(), self.energy, self.speed, heading)


class TwoStateChannel:
    """Generates the high and low channel states."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed + 42)
        self.state = 1

    def step(self) -> int:
        if self.state == 1 and self.rng.rand() < high_to_low_probability:
            self.state = 0
        elif self.state == 0 and self.rng.rand() < low_to_high_probability:
            self.state = 1
        return self.state


def create_uavs(seed: int, uav_count: int) -> List[UAV]:
    return [
        UAV(
            uav_id=uav_id,
            rng=np.random.RandomState(seed * 1000 + uav_id),
            uav_count=uav_count,
        )
        for uav_id in range(uav_count)
    ]


# Reference lifetime and reward
def get_stable_lifetime(
    environment: str,
    channel_state: int,
    position: np.ndarray,
    energy: float,
    speed: float,
    heading: float,
) -> float:
    """Returns the continuous reference lifetime for one state."""
    if environment == "Simple":
        return high_state_lifetime if channel_state == 1 else low_state_lifetime
    x, y, z = map(float, position)
    snr = high_snr if channel_state == 1 else low_snr
    if environment == "NonLinear":
        jam = 1.0 if 800.0 <= x <= 1200.0 and 800.0 <= y <= 1200.0 else 0.0
        arg = (
            3.0 * snr
            - 0.1 * speed
            - 0.005 * z
            - 0.5 * (heading / np.pi)
            + 2.0 * energy
            - 4.0 * jam
        )
        return float(TAU_MIN + (TAU_MAX - TAU_MIN) / (1.0 + np.exp(-arg)))
    if environment == "MultiModal":
        jam1 = 1.0 if 400.0 <= x <= 800.0 and 400.0 <= y <= 800.0 else 0.0
        jam2 = 1.0 if 1200.0 <= x <= 1600.0 and 1200.0 <= y <= 1600.0 else 0.0
        jam3 = 1.0 if 800.0 <= x <= 1200.0 and 1400.0 <= y <= 1800.0 else 0.0
        spatial = np.sin(2.0 * np.pi * x / X_MAX) * np.cos(2.0 * np.pi * y / Y_MAX)
        los = 1.0 / (1.0 + np.exp(-0.05 * (z - 250.0)))
        speed_norm = speed / V_MAX_UAV
        doppler = -4.0 * (speed_norm - 0.3) ** 2 + 0.5
        battery_term = 1.0 if energy >= 0.25 else -2.0
        heading_term = -2.0 * abs(np.sin(2.0 * heading))
        arg = (
            2.5 * snr
            + 1.5 * spatial
            + los
            + doppler
            + 1.5 * battery_term
            + heading_term
            - 3.0 * jam1
            - 3.5 * jam2
            - 2.5 * jam3
        )
        base = 1.0 / (1.0 + np.exp(-arg))
        modulation = (
            0.15 * np.sin(4.0 * np.pi * x / X_MAX) * np.cos(6.0 * np.pi * y / Y_MAX)
        )
        clipped = np.clip(base + modulation, 0.05, 0.95)
        return float(TAU_MIN + (TAU_MAX - TAU_MIN) * clipped)
    raise ValueError(f"Unsupported environment: {environment}")


def drop_probability(tau: float, stable_lifetime: float) -> float:
    if tau <= stable_lifetime:
        return 0.0
    return float(min(1.0, (tau - stable_lifetime) / (0.4 * TAU_MAX)))


def expected_reward(tau: float, stable_lifetime: float) -> float:
    drop_chance = drop_probability(tau, stable_lifetime)
    reward = (
        completion_weight * (1.0 - drop_chance)
        - drop_weight * drop_chance
        - exposure_weight * (tau / TAU_MAX)
        - overhead_weight * (TAU_MAX / max(tau, 1.0))
    )
    return float(np.clip(reward, -1.0, 1.0))


def best_discrete_timeout(stable_lifetime: float) -> tuple[int, int, float]:
    rewards = np.asarray(
        [expected_reward(float(tau), stable_lifetime) for tau in TAU_BINS],
        dtype=np.float64,
    )
    arm_index = int(np.argmax(rewards))
    return (arm_index, int(TAU_BINS[arm_index]), float(rewards[arm_index]))


def get_selected_environments(value: str) -> Sequence[str]:
    if value == "all":
        return ENVIRONMENTS
    if value not in ENVIRONMENTS:
        raise ValueError(
            f"ENVIRONMENT must be 'all', 'Simple', 'NonLinear', or 'MultiModal'; got {value!r}"
        )
    return (value,)


# CSV layout
def make_column_names(
    environments: Sequence[str], include_all_rewards: bool
) -> List[str]:
    fields = [
        "round",
        "seed",
        "uav_id",
        "channel_state",
        "channel_label",
        "x_m",
        "y_m",
        "z_m",
        "energy",
        "snr",
        "speed_mps",
        "phi_rad",
        "ctx_x",
        "ctx_y",
        "ctx_z",
        "ctx_energy",
        "ctx_snr",
        "ctx_speed",
        "ctx_phi",
    ]
    for env in environments:
        prefix = env.lower()
        fields.extend(
            [
                f"{prefix}_tau_star_s",
                f"{prefix}_oracle_arm_0based",
                f"{prefix}_oracle_arm_1based",
                f"{prefix}_oracle_tau_s",
                f"{prefix}_oracle_expected_reward",
            ]
        )
        if include_all_rewards:
            for tau in TAU_BINS:
                tau_int = int(tau)
                fields.extend(
                    [
                        f"{prefix}_drop_prob_tau_{tau_int:03d}s",
                        f"{prefix}_expected_reward_tau_{tau_int:03d}s",
                    ]
                )
    return fields


def make_data_row(
    seed: int,
    round_index: int,
    channel: TwoStateChannel,
    uavs: List[UAV],
    total_rounds: int,
    uav_count: int,
    environments: Sequence[str],
    include_all_rewards: bool,
) -> Dict[str, object]:
    channel_state = channel.step()
    uav_id = round_index % uav_count
    uav = uavs[uav_id]
    position, energy, speed, heading = uav.step(total_rounds)
    x, y, z = map(float, position)
    snr = high_snr if channel_state == 1 else low_snr
    row: Dict[str, object] = {
        "round": round_index + 1,
        "seed": seed,
        "uav_id": uav_id,
        "channel_state": channel_state,
        "channel_label": "high" if channel_state == 1 else "low",
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "energy": float(energy),
        "snr": float(snr),
        "speed_mps": float(speed),
        "phi_rad": float(heading),
        "ctx_x": x / X_MAX,
        "ctx_y": y / Y_MAX,
        "ctx_z": z / Z_MAX,
        "ctx_energy": float(energy),
        "ctx_snr": float(snr),
        "ctx_speed": min(speed / V_MAX_UAV, 1.0),
        "ctx_phi": heading / np.pi,
    }
    for env in environments:
        prefix = env.lower()
        stable_lifetime = get_stable_lifetime(
            environment=env,
            channel_state=channel_state,
            position=position,
            energy=energy,
            speed=speed,
            heading=heading,
        )
        oracle_arm, oracle_tau, oracle_reward = best_discrete_timeout(stable_lifetime)
        row[f"{prefix}_tau_star_s"] = stable_lifetime
        row[f"{prefix}_oracle_arm_0based"] = oracle_arm
        row[f"{prefix}_oracle_arm_1based"] = oracle_arm + 1
        row[f"{prefix}_oracle_tau_s"] = oracle_tau
        row[f"{prefix}_oracle_expected_reward"] = oracle_reward
        if include_all_rewards:
            for tau in TAU_BINS:
                tau_int = int(tau)
                row[f"{prefix}_drop_prob_tau_{tau_int:03d}s"] = drop_probability(
                    float(tau), stable_lifetime
                )
                row[f"{prefix}_expected_reward_tau_{tau_int:03d}s"] = expected_reward(
                    float(tau), stable_lifetime
                )
    return row


def format_csv_value(value: object) -> object:
    """Keeps integer values as integers and writes floats consistently."""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10f}"
    if isinstance(value, np.integer):
        return int(value)
    return value


# File generation
def write_seed_file(
    seed: int,
    output_dir: Path,
    rows: int,
    uav_count: int,
    environment: str,
    include_all_rewards: bool,
) -> Path:
    """Creates one CSV file for one random seed."""
    environments = get_selected_environments(environment)
    column_names = make_column_names(environments, include_all_rewards)
    channel = TwoStateChannel(seed)
    uavs = create_uavs(seed, uav_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"UAV Simulation for seed {seed}.csv"
    print(f"\nGenerating seed {seed}")
    print(f"File: {output_path.name}")
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(column_names)
        for round_index in range(rows):
            row = make_data_row(
                seed=seed,
                round_index=round_index,
                channel=channel,
                uavs=uavs,
                total_rounds=rows,
                uav_count=uav_count,
                environments=environments,
                include_all_rewards=include_all_rewards,
            )
            writer.writerow([format_csv_value(row[field]) for field in column_names])
            completed = round_index + 1
            if completed % progress_interval == 0 or completed == rows:
                print(
                    f"  Seed {seed}: {completed:,}/{rows:,} rows ({completed * 100.0 / rows:5.1f}%)",
                    flush=True,
                )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Completed seed {seed}: {rows:,} rows, {len(column_names)} columns, {size_mb:.2f} MB"
    )
    return output_path


def write_all_seed_files(
    output_dir: Path,
    first_seed: int,
    number_of_seeds: int,
    rows: int,
    uav_count: int,
    environment: str,
    include_all_rewards: bool,
) -> List[Path]:
    if rows <= 0:
        raise ValueError("ROUNDS_PER_SEED must be positive")
    if number_of_seeds <= 0:
        raise ValueError("NUMBER_OF_SEEDS must be positive")
    if uav_count <= 0:
        raise ValueError("NUMBER_OF_UAVS must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: List[Path] = []
    last_seed = first_seed + number_of_seeds - 1
    print("\nStarting UAV dataset generation")
    print(f"Output folder    : {output_dir}")
    print(f"Seeds            : {first_seed} through {last_seed}")
    print(f"Rounds per seed  : {rows:,}")
    print(f"UAVs per seed    : {uav_count}")
    print(f"Environment data : {environment}")
    for seed in range(first_seed, first_seed + number_of_seeds):
        output_files.append(
            write_seed_file(
                seed=seed,
                output_dir=output_dir,
                rows=rows,
                uav_count=uav_count,
                environment=environment,
                include_all_rewards=include_all_rewards,
            )
        )
    print("\nAll seed files were generated successfully:")
    for path in output_files:
        print(f"  {path.name}")
    return output_files


def mount_google_drive() -> bool:
    """Mounts Google Drive when the script is running in Colab."""
    if not use_google_drive:
        return False
    try:
        from google.colab import drive
    except ImportError:
        print(
            "Google Colab was not detected. Files will be saved under /content instead."
        )
        return False
    drive.mount("/content/drive")
    return True


def main() -> None:
    drive_mounted = mount_google_drive()
    if drive_mounted:
        output_dir = Path(drive_output_folder)
    else:
        output_dir = Path("/content/UAV Simulation Datasets")
    write_all_seed_files(
        output_dir=output_dir,
        first_seed=first_seed,
        number_of_seeds=number_of_seeds,
        rows=rounds_per_seed,
        uav_count=number_of_uavs,
        environment=selected_environment,
        include_all_rewards=save_all_arm_rewards,
    )


if __name__ == "__main__":
    main()
