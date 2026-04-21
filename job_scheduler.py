import pandas as pd
import numpy as np
import os
import heapq
import pickle

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
CHECKPOINT_DIR = "checkpoints_parallel"
CHECKPOINT_EVERY = 10_000

TOTAL_NODES = 422

# Strong SJF bias (critical for low waiting time)
ALPHA = 0.85   # predicted runtime
BETA = 0.15    # aging (small!)

MIN_RESERVE = 0.10
MAX_RESERVE = 0.20

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ======================================================
# HELPERS
# ======================================================

def parse_walltime_to_hours(wt):
    if pd.isna(wt):
        return np.nan
    wt = str(wt).strip()
    if ":" in wt:
        p = [float(x) for x in wt.split(":")]
        if len(p) == 3:
            return p[0] + p[1]/60 + p[2]/3600
        if len(p) == 2:
            return p[0]/60 + p[1]/3600
    try:
        return float(wt) / 3600
    except:
        return np.nan

def save_checkpoint(state, n):
    path = f"{CHECKPOINT_DIR}/checkpoint_{n}.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"    ✔ Checkpoint saved @ {n}")

def load_latest_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pkl")]
    if not files:
        return None
    latest = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    with open(os.path.join(CHECKPOINT_DIR, latest), "rb") as f:
        print(f"    ▶ Resuming from {latest}")
        return pickle.load(f)

# ======================================================
# 1. LOAD & CLEAN
# ======================================================

print("[1/8] Loading dataset...")
df = pd.read_csv(CSV_PATH)

df["submit_time"] = pd.to_datetime(df["qtime"], errors="coerce")
df["start_time"] = pd.to_datetime(df["stime"], errors="coerce")

df["execution_time_hours"] = df["execution_time(hours)"]

if "walltime_requested" in df.columns:
    df["requested_time_hours"] = df["walltime_requested"].apply(parse_walltime_to_hours)
else:
    df["requested_time_hours"] = df["execution_time_hours"] * 1.2

df = df[
    (df["execution_time_hours"] > 0) &
    (df["requested_time_hours"] > 0) &
    (df["nodes_requested"] > 0) &
    df["submit_time"].notna()
].copy()

print("    Jobs after cleaning:", len(df))

# ======================================================
# 2. WALLTIME PREDICTOR
# ======================================================

print("[2/8] Training walltime predictor...")

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
print(f"    ✔ MAE: {mae:.2f} hours")

df["predicted_time_hours"] = np.clip(model.predict(X), 0.05, None)

# ======================================================
# 3. PRIORITY (SJF + LIGHT AGING)
# ======================================================

df["priority"] = (
    ALPHA * df["predicted_time_hours"] +
    BETA * df["requested_time_hours"]
)

# ======================================================
# 4. RESERVATION
# ======================================================

single_ratio = (df["nodes_requested"] == 1).mean()
reserve_frac = min(MAX_RESERVE, max(MIN_RESERVE, single_ratio))
RESERVED_SINGLE = int(TOTAL_NODES * reserve_frac)

print(f"[3/8] Reserved single-node pool: {RESERVED_SINGLE}")

# ======================================================
# 5. PARALLEL DISCRETE EVENT SIMULATION
# ======================================================

print("[4/8] Starting parallel scheduler simulation...")

checkpoint = load_latest_checkpoint()

if checkpoint:
    current_time = checkpoint["current_time"]
    available_nodes = checkpoint["available_nodes"]
    future_events = checkpoint["future_events"]
    waiting = checkpoint["waiting"]
    completed = checkpoint["completed"]
else:
    current_time = df["submit_time"].min()
    available_nodes = TOTAL_NODES
    future_events = []   # (finish_time, job_id, nodes)
    waiting = df.sort_values("submit_time").to_dict("records")
    completed = []

job_counter = len(completed)

while waiting or future_events:

    # Release finished jobs
    while future_events and future_events[0][0] <= current_time:
        finish, _, nodes = heapq.heappop(future_events)
        available_nodes += nodes

    # Ready queue
    ready = [j for j in waiting if j["submit_time"] <= current_time]
    waiting = [j for j in waiting if j["submit_time"] > current_time]

    ready.sort(key=lambda j: j["priority"])

    scheduled_any = False

    for job in ready:
        nodes = job["nodes_requested"]

        if nodes == 1 and available_nodes - RESERVED_SINGLE < 1:
            continue

        if nodes <= available_nodes:
            start = current_time
            finish = start + pd.Timedelta(hours=job["predicted_time_hours"])

            heapq.heappush(
                future_events,
                (finish, job["Job_Id"], nodes)
            )

            available_nodes -= nodes
            scheduled_any = True

            completed.append({
                "Job_Id": job["Job_Id"],
                "waiting_time_hours": (
                    (start - job["submit_time"]).total_seconds() / 3600
                )
            })

            job_counter += 1

            if job_counter % CHECKPOINT_EVERY == 0:
                save_checkpoint({
                    "current_time": current_time,
                    "available_nodes": available_nodes,
                    "future_events": future_events,
                    "waiting": waiting,
                    "completed": completed
                }, job_counter)
        else:
            waiting.append(job)

    if not scheduled_any:
        if future_events:
            current_time = future_events[0][0]
        elif waiting:
            current_time = min(j["submit_time"] for j in waiting)

# ======================================================
# 6. RESULTS
# ======================================================

print("[8/8] Final results")

res = pd.DataFrame(completed)

print("    Avg waiting time (hrs):",
      round(res["waiting_time_hours"].mean(), 3))
print("    95th percentile wait:",
      round(res["waiting_time_hours"].quantile(0.95), 3))

res.to_csv("ml_fair_parallel_results.csv", index=False)
print("    ✔ Results saved")
print("\n✅ Parallel scheduler simulation complete")

