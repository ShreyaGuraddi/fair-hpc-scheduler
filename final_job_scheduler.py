import pandas as pd
import numpy as np
import heapq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = "HPC_scheduler/pbs_jobs_parsed.csv"
TOTAL_NODES = 422

ALPHA = 0.85
AGING_FACTOR = 0.05

FAIRNESS_THRESHOLD = 4
LARGE_JOB_THRESHOLD = 32

# 🔹 Conservative Cushion
LARGE_NODE_RESERVE_RATIO = 0.10
LARGE_NODE_RESERVE = int(TOTAL_NODES * LARGE_NODE_RESERVE_RATIO)

MIN_RESERVE = 0.10
MAX_RESERVE = 0.20

# ======================================================
# LOAD & CLEAN
# ======================================================

print("[1/6] Loading dataset...")
df = pd.read_csv(CSV_PATH)

df["submit_time"] = pd.to_datetime(df["qtime"], errors="coerce")
df["execution_time_hours"] = df["execution_time(hours)"]

def parse_walltime_to_hours(wt):
    if pd.isna(wt): return np.nan
    wt = str(wt)
    if ":" in wt:
        p = [float(x) for x in wt.split(":")]
        return p[0] + p[1]/60 + p[2]/3600
    return float(wt)/3600

df["requested_time_hours"] = df["walltime_requested"].apply(parse_walltime_to_hours)

df = df[
    (df["execution_time_hours"] > 0) &
    (df["requested_time_hours"] > 0) &
    (df["nodes_requested"] > 0) &
    df["submit_time"].notna()
].copy()

print("Jobs after cleaning:", len(df))

# ======================================================
# ML WALLTIME PREDICTOR
# ======================================================

print("[2/6] Training walltime predictor...")

features = ["nodes_requested", "requested_time_hours", "Queue", "User"]
X = df[features]
y = df["execution_time_hours"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Queue", "User"]),
    ("num", "passthrough", ["nodes_requested", "requested_time_hours"])
])

model = Pipeline([
    ("prep", pre),
    ("rf", RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    ))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(Xtr, ytr)

mae = mean_absolute_error(yte, model.predict(Xte))
print(f"✔ MAE: {mae:.2f} hours")

df["predicted_time_hours"] = np.clip(model.predict(X), 0.05, None)

# ======================================================
# RESERVATION
# ======================================================

single_ratio = (df["nodes_requested"] == 1).mean()
reserve_frac = min(MAX_RESERVE, max(MIN_RESERVE, single_ratio))
RESERVED_SINGLE = int(TOTAL_NODES * reserve_frac)

print(f"[3/6] Reserved single-node pool: {RESERVED_SINGLE}")
print(f"Large-job cushion: {LARGE_NODE_RESERVE} nodes")

# ======================================================
# SIMULATION
# ======================================================

print("[4/6] Starting adaptive fairness SJF simulation...")

current_time = df["submit_time"].min()
available_nodes = TOTAL_NODES

future_events = []
waiting = df.sort_values("submit_time").to_dict("records")
completed = []

while waiting or future_events:

    while future_events and future_events[0][0] <= current_time:
        finish, _, nodes = heapq.heappop(future_events)
        available_nodes += nodes

    ready = [j for j in waiting if j["submit_time"] <= current_time]
    waiting = [j for j in waiting if j["submit_time"] > current_time]

    for job in ready:
        waiting_hours = (
            (current_time - job["submit_time"]).total_seconds() / 3600
        )
        size_factor = np.log1p(job["nodes_requested"])
        base_priority = ALPHA * job["predicted_time_hours"]

        if job["nodes_requested"] >= LARGE_JOB_THRESHOLD:
            fairness_boost = 1 + (waiting_hours / FAIRNESS_THRESHOLD)
        else:
            fairness_boost = 1.0

        job["dynamic_priority"] = (
            base_priority
            - AGING_FACTOR * waiting_hours * size_factor * fairness_boost
        )

    ready.sort(key=lambda j: j["dynamic_priority"])

    scheduled_any = False

    large_jobs_waiting = any(
        j["nodes_requested"] >= LARGE_JOB_THRESHOLD for j in ready
    )

    for job in ready:
        nodes = job["nodes_requested"]

        if nodes == 1 and available_nodes - RESERVED_SINGLE < 1:
            continue

        if nodes <= available_nodes:

            # Cushion logic
            if (
                nodes < LARGE_JOB_THRESHOLD
                and large_jobs_waiting
                and available_nodes - nodes < LARGE_NODE_RESERVE
            ):
                continue

            start = current_time
            finish = start + pd.Timedelta(hours=job["predicted_time_hours"])

            heapq.heappush(future_events, (finish, job["Job_Id"], nodes))
            available_nodes -= nodes
            scheduled_any = True

            completed.append({
                "waiting_time_hours":
                (start - job["submit_time"]).total_seconds() / 3600,
                "nodes": nodes
            })

        else:
            waiting.append(job)

    if not scheduled_any:
        if future_events:
            current_time = future_events[0][0]
        elif waiting:
            current_time = min(j["submit_time"] for j in waiting)

# ======================================================
# RESULTS
# ======================================================

res = pd.DataFrame(completed)

print("Avg wait:", round(res["waiting_time_hours"].mean(),3))
print("95th percentile:", round(res["waiting_time_hours"].quantile(0.95),3))

res["job_size_group"] = pd.cut(
    res["nodes"],
    bins=[0,4,32,10000],
    labels=["Small","Medium","Large"]
)

print(res.groupby("job_size_group")["waiting_time_hours"].mean())

