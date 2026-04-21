# 🚀 Fairness-Aware Predictive Scheduling for HPC Systems

## 📌 Project Overview

This repository implements a complete experimental framework for evaluating scheduling strategies in **High Performance Computing (HPC)** environments.

It accompanies the research work:

> **“Fairness-Aware Predictive Scheduling for Production HPC Systems”**

The project focuses on improving scheduling efficiency using **machine learning** and **fairness-aware mechanisms**.

---

## 🎯 Key Features

* ✅ FCFS with EASY Backfilling (baseline)
* ✅ Machine Learning-based Scheduler (ML-SJF)
* ✅ Fairness-Aware Adaptive Scheduler (proposed)
* ✅ PBS workload parsing and dataset generation
* ✅ Comparative performance analysis

---

## 📂 Project Structure

```bash
.
├── parse_pbs_jobs.py          # Convert PBS logs → structured dataset
├── fcfs_backfilling.py        # Baseline scheduler (FCFS + EASY)
├── job_scheduler.py           # ML-based scheduler (Random Forest)
├── final_job_scheduler.py     # Fairness-aware scheduler
├── HPC_scheduler/
│   ├── qstat_output.txt       # Raw HPC logs (input)
│   └── pbs_jobs_parsed.csv    # Generated dataset
├── results/                   # Output files (generated)
└── README.md
```

---

## ⚙️ Installation

### 🔹 Requirements

* Python 3.8+
* pip

### 🔹 Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## ▶️ How to Run

Follow the pipeline **in order**:

### 1️⃣ Parse PBS Logs

```bash
python parse_pbs_jobs.py
```

---

### 2️⃣ Run FCFS + EASY Backfilling

```bash
python fcfs_backfilling.py
```

📁 Output:

```
HPC_scheduler/fcfs_backfilling_comparison.csv
```

---

### 3️⃣ Run ML-Based Scheduler

```bash
python job_scheduler.py
```

📁 Output:

```
ml_fair_parallel_results.csv
```

---

### 4️⃣ Run Fairness-Aware Scheduler

```bash
python final_job_scheduler.py
```

📊 Outputs:

* Average waiting time
* 95th percentile waiting time
* Fairness analysis by job size

---

## 📊 Results Summary

| Scheduler              | Key Behavior                                   |
| ---------------------- | ---------------------------------------------- |
| **FCFS + Backfilling** | Simple baseline, higher waiting time           |
| **ML-SJF**             | Reduces average waiting time using predictions |
| **FairScheduler**      | Improves fairness + reduces tail latency       |

---

## 🧠 Methodology

### 🔹 Machine Learning Model

* Model: Random Forest Regressor
* Task: Predict job execution time
* Features:

  * Nodes requested
  * Requested runtime
  * Queue
  * User

### 🔹 Fairness Mechanism

* Dynamic priority adjustment
* Aging factor for long-waiting jobs
* Special handling for large jobs
* Resource reservation (capacity cushion)

---

## ⚙️ Configuration

Key parameters used:

```python
TOTAL_NODES = 422
ALPHA = 0.85              # Runtime importance
AGING_FACTOR = 0.05       # Fairness adjustment
LARGE_JOB_THRESHOLD = 32  # Nodes
```

---

## 📈 Expected Outcomes

* 📉 Reduced average waiting time (ML scheduler vs FCFS)
* 📉 Reduced 95th percentile latency (FairScheduler)
* ⚖️ Improved fairness across job sizes
* 🚫 Reduced starvation of large jobs

---

## 🔁 Reproducibility

* Deterministic pipeline
* Same dataset → same results
* No external dependencies beyond listed libraries

---

## ⚠️ Notes

* Scripts assume folder:

```
HPC_scheduler/
```

* Ensure input file:

```
qstat_output.txt
```

is present before running

---


## 👩‍💻 Author

**Shreya Guraddi**

---

## 📄 License

For academic and research use.

---

## ⭐ If you found this useful

Consider starring ⭐ the repo!
