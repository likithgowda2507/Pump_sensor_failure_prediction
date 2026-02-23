"""
Synthetic Industrial Pump Sensor Dataset Generator
===================================================
Generates a realistic time-series dataset for predictive maintenance
with gradual degradation patterns, realistic sensor noise, and ML targets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Configuration ────────────────────────────────────────────────────────────
np.random.seed(42)

NUM_PUMPS = 20
START_DATE = datetime(2025, 1, 1)
MONTHS_RANGE = (8, 12)  # each pump gets 8–12 months of data
FAILURE_TYPES = ["bearing", "overheating", "seal_leak", "electrical"]

# Baseline sensor ranges (mean, std)
BASELINE = {
    "Temperature_C":      (55.0, 3.0),
    "Vibration_mm_s":     (2.5, 0.4),
    "Pressure_bar":       (6.0, 0.3),
    "Flow_rate_lpm":      (120.0, 5.0),
    "Motor_current_A":    (15.0, 0.8),
    "Voltage_V":          (400.0, 5.0),
    "Power_kW":           (7.5, 0.4),
    "Efficiency_percent": (88.0, 2.0),
    "Noise_level_dB":     (72.0, 2.0),
    "Oil_level_percent":  (92.0, 2.0),
}

# Pre-failure degradation window in hours (how many hours before failure
# degradation starts ramping up)
DEGRADATION_WINDOW = (72, 200)  # 3–8 days


def generate_pump_data(pump_id: int) -> pd.DataFrame:
    """Generate time-series data for a single pump."""

    # Duration: 8–12 months
    months = np.random.randint(*MONTHS_RANGE)
    total_hours = months * 30 * 24  # approximate hours
    timestamps = [START_DATE + timedelta(hours=i) for i in range(total_hours)]
    n = len(timestamps)

    # ── Plan failure events ──────────────────────────────────────────────
    # Target ~1–3% failure rate → each pump gets a few failures
    num_failures = max(1, int(n * np.random.uniform(0.012, 0.020) / 1))
    # But failures are events, not rows — we space them out
    num_failures = np.random.randint(2, max(3, months // 2 + 1))

    # Pick failure hours (ensure minimum spacing of degradation window)
    min_spacing = max(DEGRADATION_WINDOW) + 48
    failure_hours = []
    for _ in range(num_failures * 5):  # attempt budget
        if len(failure_hours) >= num_failures:
            break
        candidate = np.random.randint(max(DEGRADATION_WINDOW) + 10, n - 10)
        if all(abs(candidate - fh) > min_spacing for fh in failure_hours):
            failure_hours.append(candidate)
    failure_hours.sort()

    # Assign failure types
    failure_assignments = []  # list of (hour, type, degradation_start)
    for fh in failure_hours:
        ftype = np.random.choice(FAILURE_TYPES)
        deg_window = np.random.randint(*DEGRADATION_WINDOW)
        deg_start = max(0, fh - deg_window)
        failure_assignments.append((fh, ftype, deg_start))

    # ── Generate baseline sensor signals ─────────────────────────────────
    data = {}
    for col, (mean, std) in BASELINE.items():
        # Smooth daily/weekly patterns + noise
        t = np.arange(n)
        daily_cycle = np.sin(2 * np.pi * t / 24) * std * 0.3
        weekly_cycle = np.sin(2 * np.pi * t / (24 * 7)) * std * 0.15
        noise = np.random.normal(0, std * 0.15, n)
        # Slow drift over time (aging)
        drift = np.linspace(0, std * 0.3, n) if col in [
            "Temperature_C", "Vibration_mm_s", "Noise_level_dB"
        ] else np.zeros(n)
        negative_drift = np.linspace(0, -std * 0.2, n) if col in [
            "Efficiency_percent", "Oil_level_percent"
        ] else np.zeros(n)

        data[col] = mean + daily_cycle + weekly_cycle + noise + drift + negative_drift

    data = {k: v.copy() for k, v in data.items()}

    # ── Apply degradation patterns before each failure ───────────────────
    failure_event = np.zeros(n, dtype=int)
    failure_type_arr = ["none"] * n
    rul_hours = np.full(n, 9999.0)
    maintenance_flag = np.zeros(n, dtype=int)

    for fh, ftype, deg_start in failure_assignments:
        deg_len = fh - deg_start
        if deg_len <= 0:
            continue

        # Degradation ramp (exponential-ish)
        ramp = np.linspace(0, 1, deg_len) ** 2  # quadratic ramp

        for i, hour in enumerate(range(deg_start, fh)):
            if hour >= n:
                break
            intensity = ramp[i]

            if ftype == "overheating":
                data["Temperature_C"][hour] += intensity * 35  # up to +35°C
                data["Efficiency_percent"][hour] -= intensity * 15
                data["Power_kW"][hour] += intensity * 3
                data["Noise_level_dB"][hour] += intensity * 8

            elif ftype == "bearing":
                data["Vibration_mm_s"][hour] += intensity * 8  # up to +8 mm/s
                data["Noise_level_dB"][hour] += intensity * 15
                data["Temperature_C"][hour] += intensity * 10
                data["Efficiency_percent"][hour] -= intensity * 10

            elif ftype == "seal_leak":
                data["Pressure_bar"][hour] -= intensity * 3  # drops by 3 bar
                data["Flow_rate_lpm"][hour] -= intensity * 40
                data["Oil_level_percent"][hour] -= intensity * 25
                data["Efficiency_percent"][hour] -= intensity * 12

            elif ftype == "electrical":
                data["Motor_current_A"][hour] += intensity * 12  # spikes
                data["Voltage_V"][hour] += intensity * np.random.choice([-30, 30])
                data["Power_kW"][hour] += intensity * 5
                data["Temperature_C"][hour] += intensity * 8

            # RUL decreases linearly in the degradation window
            rul_hours[hour] = max(0, fh - hour)

        # Mark the failure hour
        if fh < n:
            failure_event[fh] = 1
            failure_type_arr[fh] = ftype
            rul_hours[fh] = 0

            # Also mark the degradation window rows with the failure type
            for hour in range(deg_start, fh):
                if hour < n:
                    failure_type_arr[hour] = ftype
                    rul_hours[hour] = fh - hour

            # Maintenance happens 1–6 hours after failure (repair)
            maint_start = fh + 1
            maint_end = min(n, fh + np.random.randint(4, 24))
            for mh in range(maint_start, maint_end):
                if mh < n:
                    maintenance_flag[mh] = 1
                    # Sensors recover after maintenance
                    for col, (mean, std) in BASELINE.items():
                        data[col][mh] = mean + np.random.normal(0, std * 0.2)

    # ── Cap RUL at a reasonable max for non-degrading periods ────────────
    # For rows not in any degradation window, RUL stays at 9999 — we'll
    # compute true RUL to next failure
    for idx in range(n):
        if rul_hours[idx] == 9999.0:
            # Find next failure hour
            next_fail = None
            for fh, _, _ in failure_assignments:
                if fh > idx:
                    next_fail = fh
                    break
            if next_fail is not None:
                rul_hours[idx] = next_fail - idx
            else:
                # No upcoming failure — cap at remaining hours or a large value
                rul_hours[idx] = min(n - idx, 720)  # cap at 720h (30 days)

    # ── Failure_in_6h target ─────────────────────────────────────────────
    failure_in_6h = np.zeros(n, dtype=int)
    for fh, _, _ in failure_assignments:
        for h in range(max(0, fh - 6), fh + 1):
            if h < n:
                failure_in_6h[h] = 1

    # ── Running hours total (cumulative, with small downtime gaps) ───────
    running_hours = np.cumsum(np.ones(n))
    # Add a random initial offset (pump was running before data collection)
    initial_hours = np.random.randint(500, 5000)
    running_hours = running_hours + initial_hours
    # Subtract maintenance downtime
    for mh in range(n):
        if maintenance_flag[mh]:
            running_hours[mh:] -= 1

    # ── Add outliers (~0.5%) ─────────────────────────────────────────────
    num_outliers = int(n * 0.005)
    outlier_indices = np.random.choice(n, num_outliers, replace=False)
    sensor_cols = list(BASELINE.keys())
    for idx in outlier_indices:
        col = np.random.choice(sensor_cols)
        mean, std = BASELINE[col]
        data[col][idx] = mean + np.random.choice([-1, 1]) * np.random.uniform(4, 6) * std

    # ── Build DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Pump_ID": f"PUMP_{pump_id:03d}",
        "Temperature_C": np.round(data["Temperature_C"], 2),
        "Vibration_mm_s": np.round(data["Vibration_mm_s"], 2),
        "Pressure_bar": np.round(data["Pressure_bar"], 2),
        "Flow_rate_lpm": np.round(data["Flow_rate_lpm"], 2),
        "Motor_current_A": np.round(data["Motor_current_A"], 2),
        "Voltage_V": np.round(data["Voltage_V"], 2),
        "Power_kW": np.round(data["Power_kW"], 2),
        "Efficiency_percent": np.round(data["Efficiency_percent"], 2),
        "Noise_level_dB": np.round(data["Noise_level_dB"], 2),
        "Oil_level_percent": np.round(data["Oil_level_percent"], 2),
        "Running_hours_total": running_hours.astype(int),
        "Maintenance_flag": maintenance_flag,
        "Failure_event": failure_event,
        "Failure_type": failure_type_arr,
        "RUL_hours": np.round(rul_hours, 1),
        "Failure_in_6h": failure_in_6h,
    })

    # ── Introduce missing values (~1–2%) ─────────────────────────────────
    missing_rate = np.random.uniform(0.01, 0.02)
    num_missing = int(n * missing_rate * len(sensor_cols))
    for _ in range(num_missing):
        row = np.random.randint(0, n)
        col = np.random.choice(sensor_cols)
        df.loc[row, col] = np.nan

    return df


def main():
    print("=" * 60)
    print("  Synthetic Pump Sensor Dataset Generator")
    print("=" * 60)

    all_dfs = []
    for pump_id in range(1, NUM_PUMPS + 1):
        print(f"  Generating data for PUMP_{pump_id:03d}...", end=" ")
        df = generate_pump_data(pump_id)
        all_dfs.append(df)
        print(f"{len(df):,} rows")

    dataset = pd.concat(all_dfs, ignore_index=True)

    # Sort by Pump_ID then Timestamp (chronological per pump)
    dataset.sort_values(["Pump_ID", "Timestamp"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # ── Summary statistics ───────────────────────────────────────────────
    total_rows = len(dataset)
    failure_count = dataset["Failure_event"].sum()
    failure_rate = failure_count / total_rows * 100
    missing_pct = dataset.isnull().sum().sum() / (total_rows * len(dataset.columns)) * 100

    print(f"\n{'─' * 60}")
    print(f"  Total rows:          {total_rows:,}")
    print(f"  Pumps:               {dataset['Pump_ID'].nunique()}")
    print(f"  Date range:          {dataset['Timestamp'].min()} → {dataset['Timestamp'].max()}")
    print(f"  Failure events:      {failure_count} ({failure_rate:.2f}%)")
    print(f"  Missing values:      {missing_pct:.2f}%")
    print(f"  Failure types:       {dataset[dataset['Failure_event']==1]['Failure_type'].value_counts().to_dict()}")
    print(f"{'─' * 60}")

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "pump_sensor_data.csv")
    dataset.to_csv(output_path, index=False)
    print(f"\n  ✅ Dataset saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
