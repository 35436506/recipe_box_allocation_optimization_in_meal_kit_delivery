# Box Allocation Optimization in Meal Kit Delivery

This project introduces the **Box Allocation Problem (BAP)**, a novel optimisation challenge within the UK meal kit delivery industry. Faced with complex operational demands, companies must efficiently assign customer orders across multiple production facilities over a planning horizon. This research provides a data-driven framework to guide these decisions, aiming to stabilise daily production, reduce food waste, and improve ingredient forecasting.

At the core of this project is a **Mixed-Integer Linear Programming (MILP)** model that minimises day-to-day variation in recipe allocations. The model is solved using the **open-source CBC solver** and systematically benchmarked against heuristic methods, including **Tabu Search (TS)** and **Iterative Targeted Pairwise Swap (ITPS)**.

This case study demonstrates how exact optimisation methods can provide highly efficient and robust solutions for complex, large-scale supply chain challenges, **outperforming common heuristics in both solution quality and computational time**.

---

## Manuscript

The LaTeX source files for the manuscript can be accessed and viewed on Overleaf:  
[Manuscript on Overleaf](https://www.overleaf.com/read/kkhtrxzdmtfd#9f0ecc)

---

## Repository Structure

### **01_Benchmark test (B&B, ITPS, TS)**
Scripts to reproduce the benchmark comparison of the CBC (B&B) solver, ITPS, and TS heuristics on a 10,000-order instance.

### **02_Iteration test (ITPS, TS)**
Code used to determine the optimal number of iterations for the ITPS and TS heuristic methods.

### **03_Scalability test (B&B, ITPS, TS)**
Scripts to evaluate and compare the performance and optimisation time of all three methods across varying problem sizes (10,000–100,000 orders).

### **04_Temporal fixed test (B&B, TS)**
Runs the temporal analysis over a 15-day planning horizon under stable, idealised conditions (fixed capacity and orders).

### **05_Temporal variation test (B&B, TS)**
Scripts to reproduce the temporal analysis under dynamic, real-world conditions, including sudden changes in factory capacity and daily modifications to customer orders.

### **Figures**
All figures included in the article.

---

## Requirements to Run Code

- **Python** ≥ 3.8  
- **COIN-OR Branch and Cut (CBC) Solver** — Open-source MILP solver  
- **Pyomo** / **PuLP** — Optimisation modelling libraries  
- **pandas**, **NumPy** — Data handling and manipulation  
- **Matplotlib** — Generating plots and figures  
