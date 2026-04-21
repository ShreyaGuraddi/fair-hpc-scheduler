import re
import csv
from datetime import datetime, timedelta

# -------------------------------
# SAFE TIME PARSER
# -------------------------------
def parse_time(t):
    if not t:
        return None
    try:
        return datetime.strptime(t.strip(), "%a %b %d %H:%M:%S %Y")
    except:
        return None

def parse_walltime(w):
    if not w:
        return None
    try:
        h, m, s = map(int, w.split(":"))
        return timedelta(hours=h, minutes=m, seconds=s)
    except:
        return None

# ----------------------------------------
# PARSE FULL LOG FILE (PBS FORMAT)
# ----------------------------------------
def parse_pbs_log(filename):
    jobs = []
    current = {}

    with open(filename, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            # New job begins
            if line.startswith("Job Id:"):
                if current:
                    jobs.append(current)
                jid = line.split("Job Id:")[1].strip()
                current = {"Job_Id": jid}
                continue

            # Key = Value format
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                current[key] = value

    if current:
        jobs.append(current)

    return jobs


# -----------------------------------------------------------
# TRANSFORM RAW JOB DATA → CLEAN CSV-FRIENDLY FORMAT
# -----------------------------------------------------------
def transform(jobs):
    clean = []

    for j in jobs:
        # extract standard fields safely
        jid = j.get("Job_Id", "")
        name = j.get("Job_Name", "")
        owner = j.get("Job_Owner", "")
        queue = j.get("queue", "")
        state = j.get("job_state", "")

        # Timestamps
        ctime = parse_time(j.get("ctime"))
        qtime = parse_time(j.get("qtime"))
        stime = parse_time(j.get("stime"))
        wall = parse_walltime(j.get("resources_used.walltime"))

        # Compute times
        waiting = (stime - qtime).total_seconds()/3600 if (stime and qtime) else None
        execution = wall.total_seconds()/3600 if wall else None
        turnaround = (stime - ctime).total_seconds()/3600 if (stime and ctime) else None

        clean.append({
            "Job_Id": jid,
            "Job_Name": name,
            "User": owner.split("@")[0] if "@" in owner else owner,
            "Queue": queue,
            "State": state,
            "ctime": ctime,
            "qtime": qtime,
            "stime": stime,
            "walltime(hh:mm:ss)": j.get("resources_used.walltime", ""),

            "waiting_time(hours)": waiting,
            "execution_time(hours)": execution,
            "turnaround_time(hours)": turnaround,
        })

    return clean


# ------------------------------------
# SAVE CLEAN CSV
# ------------------------------------
def save_csv(jobs, out_csv):
    fieldnames = list(jobs[0].keys())

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(jobs)

    print(f"✅ CSV saved as {out_csv}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    txt_file = "HPC_scheduler/qstat_output.txt"
    out_csv = "HPC_scheduler/pbs_jobs_parsed.csv"

    print("📥 Reading PBS job logs...")
    raw = parse_pbs_log(txt_file)
    print(f"Found {len(raw)} jobs.")

    print("🔧 Transforming...")
    jobs = transform(raw)

    print("📤 Saving CSV...")
    save_csv(jobs, out_csv)

    print("🎉 Done!")

