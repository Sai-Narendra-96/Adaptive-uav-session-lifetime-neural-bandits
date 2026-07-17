# Data files

This folder contains 20 CSV files, one for each random seed.

The file names are:

```text
UAV Simulation for seed 0.csv
UAV Simulation for seed 1.csv
...
UAV Simulation for seed 19.csv
```

Each file has 30,000 rows. One row represents one authentication round in the
simulation.

## Main columns

- `round`: simulation round number
- `seed`: random seed used for the file
- `uav_id`: UAV selected in that round
- `channel_state`: 1 for the high channel state and 0 for the low channel state
- `x_m`, `y_m`, `z_m`: UAV position in metres
- `energy`: remaining energy value used by the simulator
- `snr`: normalized signal-to-noise ratio
- `speed_mps`: UAV speed in metres per second
- `phi_rad`: heading value in radians

The columns starting with `ctx_` are the seven normalized inputs given to the
contextual bandit methods.

For each channel environment, the file also contains:

- the continuous reference session lifetime
- the best arm number
- the best timeout from 10 to 200 seconds
- the expected reward for that timeout

The three environment prefixes are `simple`, `nonlinear` and `multimodal`.

These files contain synthetic simulation data. They do not contain personal
information or data from real users.
