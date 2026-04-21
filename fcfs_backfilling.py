#!/usr/bin/env python3
"""
fcfs_easy_backfilling_compare.py

Single-queue FCFS + EASY Backfilling scheduler:
- Preserves FCFS order
- Head job gets reservation
- Backfilling allowed only if it does NOT delay head job
- Uses actual execution time when available
- Outputs CSV comparison with original waits
"""

import pandas as pd
import heapq
import csv
from datetime import datetime
import math

# ---------------- CONFIG ----------------
CSV_PATH = "HPC_scheduler/pbs_jobs_parsed.csv"
OUT_CSV  = "HPC_scheduler/fcfs_backfilling_comparison.csv"
TOTAL_NODES = 422
FALLBACK_EXEC_S = 3600
# ---------------------------------------


# ---------------- Utilities ----------------
def parse_timestamp_to_int(x):
    if pd.isna(x):
        return None
    ts = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(ts) else int(ts.timestamp())


def parse_walltime_to_seconds(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if ":" in s:
        parts = list(map(int, s.split(":")))
        if len(parts) == 3:
            h, m, sec = parts
            return h*3600 + m*60 + sec
        if len(parts) == 4:
            d, h, m, sec = parts
            return d*86400 + h*3600 + m*60 + sec
    try:
        return int(float(s) * 3600)
    except:
        return None


# ---------------- Load Jobs ----------------
def load_jobs(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    for col in ("ctime", "qtime", "stime", "execution_end_time"):
        df[col+"_s"] = df[col].apply(parse_timestamp_to_int) if col in df.columns else None

    df["arrival_s"] = df[["qtime_s","ctime_s","stime_s"]].bfill(axis=1).iloc[:,0]

    def exec_s(r):
        st = r.get("stime_s")
        et = r.get("execution_end_time_s")

        # 1️⃣ Actual execution time (preferred)
        if pd.notna(st) and pd.notna(et):
            try:
                st_i = int(st)
                et_i = int(et)
                if et_i > st_i:
                    return et_i - st_i
            except:
                pass

        # 2️⃣ execution_time(hours) column fallback
        if "execution_time(hours)" in r and pd.notna(r["execution_time(hours)"]):
            try:
                return int(float(r["execution_time(hours)"]) * 3600)
            except:
                pass

        # 3️⃣ Final fallback
        return FALLBACK_EXEC_S


    df["exec_s"] = df.apply(exec_s, axis=1)

    if "nodes_requested" in df.columns:
        df["nodes_req"] = df["nodes_requested"].astype(float).fillna(1).astype(int)
    else:
        df["nodes_req"] = 1

    def orig_wait(r):
        arr = r.get("arrival_s")
        st  = r.get("stime_s")

        if pd.notna(arr) and pd.notna(st):
            try:
                return (int(st) - int(arr)) / 3600.0
            except:
                return None
        return None


    df["orig_wait_h"] = df.apply(orig_wait, axis=1)

    jobs = []
    for i,r in df.iterrows():
        if pd.isna(r["arrival_s"]):
            continue
        jobs.append({
            "idx": i,
            "Job_Id": r.get("Job_Id",""),
            "Job_Name": r.get("Job_Name",""),
            "User": r.get("User",""),
            "arrival_s": int(r["arrival_s"]),
            "exec_s": int(r["exec_s"]),
            "nodes_req": int(r["nodes_req"]),
            "orig_wait_h": r["orig_wait_h"]
        })

    jobs.sort(key=lambda x: x["arrival_s"])
    return jobs


# ---------------- FCFS + EASY Backfilling ----------------
def simulate_fcfs_easy_backfilling(jobs, total_nodes):
    queue = [j.copy() for j in jobs]
    running = []   # (end_time, tie, job_idx, nodes)
    free_nodes = total_nodes
    results = {}
    tie = 0
    now = queue[0]["arrival_s"]

    while queue or running:

        # free completed jobs
        while running and running[0][0] <= now:
            end, _, _, nodes = heapq.heappop(running)
            free_nodes += nodes

        # If no queued jobs remain, advance time only if something is running
        if not queue:
            if running:
                now = running[0][0]
                continue
            else:
                break


        head = queue[0]

        # compute reservation time for head job
        if head["nodes_req"] <= free_nodes:
            reservation_time = now
        else:
            needed = head["nodes_req"] - free_nodes
            tmp_nodes = free_nodes
            tmp_time = now
            for end,_,_,nodes in sorted(running):
                tmp_nodes += nodes
                tmp_time = end
                if tmp_nodes >= head["nodes_req"]:
                    reservation_time = tmp_time
                    break

        scheduled = False

        # Try backfilling jobs (excluding head)
        for j in queue[1:]:
            if j["arrival_s"] > now:
                continue
            if j["nodes_req"] <= free_nodes:
                if now + j["exec_s"] <= reservation_time:
                    tie += 1
                    heapq.heappush(
                        running,
                        (now + j["exec_s"], tie, j["idx"], j["nodes_req"])
                    )
                    free_nodes -= j["nodes_req"]
                    results[j["idx"]] = {
                        "fcfs_wait_h": (now - j["arrival_s"]) / 3600
                    }
                    queue.remove(j)
                    scheduled = True
                    break

        # schedule head job if possible
        if not scheduled and head["arrival_s"] <= now and head["nodes_req"] <= free_nodes:
            tie += 1
            heapq.heappush(
                running,
                (now + head["exec_s"], tie, head["idx"], head["nodes_req"])
            )
            free_nodes -= head["nodes_req"]
            results[head["idx"]] = {
                "fcfs_wait_h": (now - head["arrival_s"]) / 3600
            }
            queue.pop(0)
            continue

        # advance time
        next_events = []
        if running:
            next_events.append(running[0][0])
        future_arrivals = [j["arrival_s"] for j in queue if j["arrival_s"] > now]
        if future_arrivals:
            next_events.append(min(future_arrivals))
        future = [t for t in next_events if t > now]

        if future:
            now = min(future)
        else:
            break


    # prepare output
    out = []
    for j in jobs:
        orig = j["orig_wait_h"]
        fcfs = results.get(j["idx"],{}).get("fcfs_wait_h")
        diff = orig - fcfs if orig is not None and fcfs is not None else None
        out.append({
            "Job_Id": j["Job_Id"],
            "Job_Name": j["Job_Name"],
            "User": j["User"],
            "nodes_req": j["nodes_req"],
            "original_wait_hours": f"{orig:.6f}" if orig is not None else "",
            "fcfs_wait_hours": f"{fcfs:.6f}" if fcfs is not None else "",
            "wait_difference_hours": f"{diff:.6f}" if diff is not None else ""
        })
    return out


# ---------------- Main ----------------
def main():
    jobs = load_jobs(CSV_PATH)
    rows = simulate_fcfs_easy_backfilling(jobs, TOTAL_NODES)

    keys = ["Job_Id","Job_Name","User","nodes_req",
            "original_wait_hours","fcfs_wait_hours","wait_difference_hours"]

    with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("✅ FCFS + EASY Backfilling simulation completed")
    print("📁 Output:", OUT_CSV)


if __name__ == "__main__":
    main()

