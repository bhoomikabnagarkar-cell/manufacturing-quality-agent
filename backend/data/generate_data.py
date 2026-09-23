"""
Generate simulated manufacturing process dataset.
Run once to create manufacturing_data.csv
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

START_TIME = datetime(2024, 1, 1, 8, 0, 0)
INTERVAL_MINUTES = 5
ROWS = 200

# Normal operating ranges
NORMAL = {
    "temperature": (75, 95),      # °C
    "pressure": (4.5, 5.5),       # bar
    "vibration": (0.5, 1.5),      # mm/s
    "machine_speed": (950, 1050), # RPM
    "production_rate": (45, 55),  # units/min
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


rows = []
for i in range(ROWS):
    ts = START_TIME + timedelta(minutes=INTERVAL_MINUTES * i)

    # Inject anomalies roughly every 20-25 records
    anomaly = (i % 22 == 0) or (i % 37 == 0)
    critical = anomaly and (i % 44 == 0)

    if critical:
        temperature    = round(random.uniform(105, 125), 2)
        pressure       = round(random.uniform(6.5, 8.0), 2)
        vibration      = round(random.uniform(3.5, 5.5), 2)
        machine_speed  = round(random.uniform(1200, 1400), 1)
        production_rate= round(random.uniform(30, 40), 2)
        quality_status = "Critical"
        defect         = 1
    elif anomaly:
        temperature    = round(random.uniform(97, 105), 2)
        pressure       = round(random.uniform(5.6, 6.5), 2)
        vibration      = round(random.uniform(2.0, 3.5), 2)
        machine_speed  = round(random.uniform(1060, 1200), 1)
        production_rate= round(random.uniform(40, 44), 2)
        quality_status = "Warning"
        defect         = 1 if random.random() < 0.6 else 0
    else:
        temperature    = round(random.uniform(*NORMAL["temperature"]), 2)
        pressure       = round(random.uniform(*NORMAL["pressure"]), 2)
        vibration      = round(random.uniform(*NORMAL["vibration"]), 2)
        machine_speed  = round(random.uniform(*NORMAL["machine_speed"]), 1)
        production_rate= round(random.uniform(*NORMAL["production_rate"]), 2)
        quality_status = "Normal"
        defect         = 0

    rows.append({
        "timestamp":      ts.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature":    temperature,
        "pressure":       pressure,
        "vibration":      vibration,
        "machine_speed":  machine_speed,
        "production_rate":production_rate,
        "quality_status": quality_status,
        "defect":         defect,
    })

FIELDNAMES = ["timestamp", "temperature", "pressure", "vibration",
              "machine_speed", "production_rate", "quality_status", "defect"]

with open("manufacturing_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {ROWS} rows -> manufacturing_data.csv")
