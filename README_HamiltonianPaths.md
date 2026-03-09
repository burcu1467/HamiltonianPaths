# 🔢 Hamiltonian Path Analysis on Total Graphs of Z_n Rings

A research tool for analyzing algebraic and graph-theoretic properties of **Total Graphs T(Γ(R))** over commutative rings **Z_n**, with a focus on Hamiltonian path and cycle detection.

---

## 📐 Mathematical Background

This project investigates the structure of **Total Graphs** constructed from commutative rings Z_n, where:

- **Z(R)** — Zero-divisors of the ring Z_n
- **Reg(R)** — Regular (non-zero-divisor) elements
- **T(Γ(R))** — Total graph: vertices are all elements of Z_n, edges connect x and y if x + y is a zero-divisor

The tool verifies key theorems, including:

> **Theorem 7.2.1:** If Z(R) is an ideal, then the Total Graph T(Γ(R)) is disconnected.

---

## ✨ Features

- 🔍 **Interactive Analysis** — Analyze a single ring Z_n step by step
- ⚡ **Batch Processing** — Analyze all rings Z_n for n ∈ [2, 30] in parallel
- 🧮 **Graph Properties** — Computes zero-divisors, regular elements, connectivity, and degree distribution
- 🛤️ **Hamiltonian Detection** — Detects Hamiltonian paths and cycles using backtracking
- 📐 **Dirac's Condition** — Verifies whether the graph satisfies Dirac's theorem for Hamiltonian cycles
- 📊 **Visualizations** — Animated Hamiltonian path, adjacency matrix heatmap, degree distribution charts
- 📄 **Report Generation** — Exports HTML reports and CSV datasets automatically
- ✅ **Theorem Verification** — Automatically verifies Theorem 7.2.1 across all analyzed rings

---

## 🖼️ Screenshots

![Hamiltonian Path Visualization](screenshots/hamiltonian_graph.png)
![Adjacency Matrix Heatmap](screenshots/adjacency_heatmap.png)
![Batch Analysis Results](screenshots/batch_results.png)
![Detailed Results Table](screenshots/results_table.png)
![HTML Report](screenshots/html_report.png)

---

## 🏗️ Tech Stack

| Purpose | Library |
|---------|---------|
| Graph construction & analysis | `networkx` |
| Visualizations & animations | `matplotlib`, `seaborn` |
| Data processing | `numpy`, `pandas` |
| Parallel processing | `multiprocessing` |
| Progress tracking | `tqdm` |
| Report generation | HTML (via Python) |
| Data export | CSV |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/burcu1467/HamiltonianPaths.git
cd HamiltonianPaths

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
python ring_analysis.py
```

You will be prompted to choose a mode:

```
Ring Analysis: Total Graph of Z_n
==========================================
Choose mode:
1. Interactive Analysis (single n)
2. Batch Processing (n in range [2, 31])

Enter your choice (1 or 2):
```

**Interactive mode** — Enter a single value of n to get full analysis, visualizations, and an HTML report.

**Batch mode** — Analyzes all rings Z_n for n ∈ [2, 30] using parallel processing and generates a comprehensive report.

---

## 📊 Sample Results (n = 10)

```
Zero-Divisors Z(R): [0, 2, 4, 5, 6, 8]
Regular Elements Reg(R): [1, 3, 7, 9]
Graph Connected: True
Z(R) is an ideal: False
Hamiltonian Path found: [0, 2, 3, 1, 4, 6, 8, 7, 5, 9]

Dirac's Condition: δ(G) ≥ n/2
Minimum degree is 5. Since n/2 is 5.0,
Dirac's condition is SATISFIED.
```

---

## 📈 Batch Analysis Summary (n ∈ [2, 30])

| Metric | Result |
|--------|--------|
| Total Rings Analyzed | 29 |
| Z(R) is Ideal | 16 (55.2%) |
| Connected Graphs | 13 (44.8%) |
| Hamiltonian Paths Found | 10 (34.5%) |
| Hamiltonian Cycles Found | 1 (3.4%) |
| Dirac Condition Satisfied | 11 rings |

---

## 📬 Contact

**Burcu** — [GitHub](https://github.com/burcu1467)
