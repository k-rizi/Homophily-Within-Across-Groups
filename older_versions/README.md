# Homophily Within and Across Groups

This repository contains the codebase for the paper "Homophily Within and Across Social Groups" by Abbas K. Rizi, Riccardo Michielan, Clara Stegehuis, and Mikko Kivelä. The paper presents a framework that integrates both local and global homophily in social networks using Exponential Random Graph Models (ERGMs). This approach distinguishes between strong homophily within tightly knit groups and weak homophily spanning broader community interactions, providing deeper insights into network dynamics, percolation thresholds, and their implications for public health and information diffusion.

The code in this repository is used to reproduce the main results of the paper, including modeling homophily with different clique sizes, generating synthetic networks, and analyzing the percolation dynamics influenced by varying levels of homophily.

If you use this code or ideas, please cite: https://arxiv.org/abs/2412.07901
K. Rizi, Abbas, et al. "Homophily Within and Across Groups." arXiv:2412.07901 (2024).


## License
This project is licensed under the MIT License - see the LICENSE file for details.


## 1) What this repository contains
- **Essential, minimal Python code** for the model, simulations, and figures:
  - `max_ent_Fig_1e.py` – maximum-entropy model toy example (Fig. 1e).
  - `Percolation_Fig3_abc.py`, `Percolation_Fig3_efg.py` – percolation experiments (Fig. 3 panels).
  - `vaccination.py` – vaccination/uptake experiments.
  - `plot_single_bar.py` – small plotting utility.
  - `code_to_generate_network.py` – helper for synthetic networks.
- **Demo data (Last.fm)** for a quick, runnable example:
  - `lastfm.edg` (edge list), `lastfm_genders.txt` (node attributes).
  - `last_fm_Fig2.ipynb` reproduces the Last.fm visualization (Fig. 2-style demo).
- A **package report** helper (optional): `pkg_report.py` (generate package/version tables for your environment).

> We provide **only essential code** with focused comments; the intended usage and rationale are described in the manuscript.  

---

## 2) System requirements
- **OS:** Linux, macOS, or Windows
- **Python:** 3.9–3.12 (tested on typical desktop/laptop)
- **Dependencies (pip):**
  - `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`, `tqdm`, `jupyter`
- **Hardware:** No non-standard hardware required.  
- **Typical install time on a normal desktop:** ≈ 2–5 minutes.

Install (recommended):
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy networkx matplotlib pandas tqdm jupyter
```

---

## 3) Data availability & scope
**Datasets used in the study were sourced from previously published research.**  
Comprehensive descriptions of **data collection protocols**, **network specifications**, and **attribute distributions** are provided in the original publications; please **refer to the cited references** for full details on each dataset.  

For this repository we **only redistribute a small Last.fm example** (edge list + gender labels) together with the visualization notebook, so users can quickly verify the pipeline end-to-end. All other datasets should be obtained from their original sources.

---

## 4) Quick start (demo)

### 4.1 Run the Last.fm notebook (recommended)
```bash
jupyter notebook last_fm_Fig2.ipynb
```
- **Expected output:** a plot showing group composition and homophily patterns for the Last.fm graph (similar to Fig. 2 in the paper).
- **Expected runtime:** < 1 minute on a normal desktop.

### 4.2 Reproduce figure examples from the scripts
```bash
# Fig. 1e toy example
python max_ent_Fig_1e.py

# Percolation experiments (Fig. 3 panels)
python Percolation_Fig3_abc.py
python Percolation_Fig3_efg.py

# Vaccination / uptake experiment
python vaccination.py
```
**Expected runtime:** typically 1–5 minutes per script on a normal desktop (depends on N and parameter sweeps).

---

## 5) How to run on **your** data
1. **Prepare inputs**
   - **Edge list:** plain text or CSV with two columns `(u, v)` (0- or 1-indexed is fine; be consistent).
   - **Node attributes:** a text/CSV file mapping `node_id -> group/attribute(s)` (e.g., `gender`).
2. **Edit the script headers** (top of `Percolation_*` and `vaccination.py`):
   - Set `EDGE_PATH`, `ATTR_PATH`, and any sampling/percolation parameters.
3. **Run**
   ```bash
   python Percolation_Fig3_abc.py
   python vaccination.py
   ```
4. **Outputs**
   - Figures (PDF/PNG) and CSV summaries are written under `./figures/` and/or `./output/` (paths noted in each script).

---

## 6) Reproduction guidance (manuscript-level)
We encourage reproducing at least one figure using the provided scripts and the Last.fm example.  
For full-scale reproductions, obtain the original datasets from their sources, set file paths as above, and run the percolation and vaccination scripts with the manuscript’s parameters.

Optional: capture your environment for the submission package:
```bash
# Everything installed (full):
pip freeze > requirements.txt

# Only the packages actually used in your current session:
python pkg_report.py --mode imported
```

---

## 7) Pseudocode (manuscript-aligned)

### 7.1 Maximum-entropy multi-scale homophily model (toy sampler)
```text
Input:
  - Groups g(i) for nodes i
  - Set of clique sizes C = {2, 3, ..., c_max}
  - Parameters θ_c for within-group preference at each social scale c (clique size)
  - Target average degree ⟨k⟩ (no hard degree constraints)

Goal:
  Sample a random graph G that maximizes entropy subject to expected multi-scale homophily.

Sketch:
  initialize G with N nodes and no edges
  repeat until average_degree(G) ≈ ⟨k⟩:
    propose an edge e = (i, j) uniformly at random
    compute Δ = Σ_{c∈C} θ_c * 𝟙{ i and j share the same c-scale group context }
           # the indicator is 1 if i and j are within the same group at level c
    accept e with probability p = logistic(Δ) = 1 / (1 + exp(-Δ))
    if accepted: add e to G
Output: G
Notes:
  - This is an exponential-family construction. The logistic acceptance implements
    expected sufficient statistics for c-scale within-group ties.
  - The stochastic block model is recovered when C={2} (pairwise) with group-specific θ_2.
```

### 7.2 Bond percolation & susceptibility-based threshold
```text
Input:
  - Graph G, group mapping g(i)
  - Edge-retention probabilities p_rr, p_rb, p_bb by endpoint groups (or a scalar p)
  - Monte Carlo repeats R

For each parameter setting:
  for r in 1..R:
    sample G_r by keeping each edge (i, j) with probability p_{g(i), g(j)}
    compute component sizes {s_1, s_2, ...} of G_r
    record largest component fraction S_r = max(s)/N
    record susceptibility χ_r = (Σ s^2 / N) - (max(s)^2 / N)   # standard definition
  aggregate:
    ⟨S⟩ = mean_r S_r
    ⟨χ⟩ = mean_r χ_r
Find threshold as the parameter value where ⟨χ⟩ attains its global maximum.
```

### 7.3 Vaccination / uptake heterogeneity
```text
Input:
  - Graph G, groups g(i)
  - Group-specific uptake u_g (probability a node in group g is vaccinated)
Process:
  for each node i:
    vaccinated[i] ~ Bernoulli(u_{g(i)})
  remove vaccinated nodes (or their incident edges) from G
  run percolation (as above) on the residual graph to estimate outbreak sizes and thresholds.
```

### 7.4 Last.fm demo pipeline
```text
Load edge list E and node attributes A (gender).
Compute group counts and mixing matrix M (edges by (group_a, group_b)).
Estimate homophily metrics (within/across) at selected scales (pairwise; cliques optional).
Plot composition and homophily curves; save figure.
```

---

## 8) Instructions for use (what the scripts do)
- **`max_ent_Fig_1e.py`**: Minimal sampler and plot demonstrating how multi-scale homophily parameters shape edge formation (toy N; deterministic seed documented in the script).
- **`Percolation_Fig3_abc.py`, `Percolation_Fig3_efg.py`**: Grid-sweep percolation with group-dependent retention; outputs ⟨S⟩, ⟨χ⟩ and figure panels.
- **`vaccination.py`**: Assigns heterogeneous uptake by group; evaluates connectivity and thresholds under removal of vaccinated nodes.
- **`last_fm_Fig2.ipynb`**: Reads `lastfm.edg` + `lastfm_genders.txt`; computes/plots group structure & homophily.

Each script has a short **“Parameters”** block at the top; edit paths and N/sweep ranges there.

---

**Contact:** Open a GitHub issue for questions or email the corresponding author.
