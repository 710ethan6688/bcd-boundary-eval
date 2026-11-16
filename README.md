# When Do Boundary Rankings Change in Building Change Detection?

Official Python verification tools, evaluation scripts, and empirical count records accompanying the manuscript:
> **"When Do Boundary Rankings Change in Building Change Detection?"**  
> *Under Review* (Submitted to IEEE Geoscience and Remote Sensing Letters)  
> **Author:** Yunhan Yang (School of Electronics and Communication Engineering, Sun Yat-sen University)

---

## 1. Overview

This repository provides a self-contained, lightweight implementation of the **Operational Directional Consistency Check ($G_{\min} > 0$)** formalized in Algorithm 1 of the paper. It evaluates whether a claimed boundary improvement between two building change detection models is directionally consistent across declared physical tolerances, scoring boundary frames, and sample omission subsets, rather than a sample-dependent artifact.

The repository includes:
1. **`boundary_claim_acceptance.py`**: Implementation of Algorithm 1 (directional gate $G_{\min} > 0$, numerical tolerance $\epsilon = 10^{-12}$, and full-set regional/object guardrails).
2. **`rscd_metrics.py`**: Region IoU, F1, and multi-tolerance Boundary F1 (BF1) calculation routines.
3. **`check_claims.py`**: Command-line verification script reproducing Table III of the main manuscript.
4. **`plot_fig2_eligibility.py`**: Vector reproduction plotting script for Fig. 2 (Heatmap and Aggregation comparisons).
5. **`data/`**: Frozen count matrices and paired evaluation logs across LEVIR-CD and WHU-CD benchmarks.

---

## 2. Quick Start

### Installation

Requires Python 3.8+:

```bash
git clone https://github.com/710ethan6688/bcd-boundary-eval.git
cd bcd-boundary-eval
pip install -r requirements.txt
```

### Reproducing Direction Checks (Table III)

Run the automated checker on the frozen 387-condition LEVIR-CD dataset:

```bash
python check_claims.py
```

Expected output:
```text
================================================================================
Comparison                     | Conditions | G_min (pp)   | Result    
--------------------------------------------------------------------------------
FC-Siam-Diff → BIT             | 387        | +7.25        | PASSED    
FC-Siam-Diff → AYANet          | 387        | +8.65        | PASSED    
FC-Siam-Diff → EGPNet          | 387        | +8.00        | PASSED    
BIT → AYANet                   | 387        | +1.32        | PASSED    
BIT → EGPNet                   | 387        | +0.64        | PASSED    
AYANet → EGPNet                | 387        | -2.20        | FAILED    
================================================================================
```

### Reproducing Figure 2

Generate Fig. 2(a) and 2(b) directly from the experimental counts:

```bash
python plot_fig2_eligibility.py
```

This exports `fig2_eligibility_aggregation.pdf` and `.png`.

---

## 3. Data Description (`data/`)

- `LEVIR_4models_counts.json`: Contains paired integer contour-matching counts, region counts, and object counts across 128 test images and three tolerances ($\theta \in \{0.5, 1.0, 2.0\}$ m) for FC-Siam-Diff, BIT, AYANet, and EGPNet.
- `LEVIR_4models_main.csv`: Benchmark summary scores on LEVIR-CD (Table II in main text).
- `WHU_342tiles_threshold_tol.csv`: Paired differences between Siamese and Standard U-Nets on 342 tiles across $\theta$ and object IoU matching thresholds $\tau \in \{0.25, 0.50, 0.75\}$ (Fig. 2(a)).
- `WHU_692objs_aggregation.csv`: Pooled contour counts vs object-equal aggregation on the identical 692 objects at $\tau = 0.75$ (Fig. 2(b)).
- `WHU_spatial_blocks_omission.csv`: Leave-one-block-out evaluations across 38 spatial blocks ($2048 \times 2048$ pixels).

---

## 4. Citation

```bibtex
@unpublished{yang2026boundary,
  title={When Do Boundary Rankings Change in Building Change Detection?},
  author={Yang, Yunhan},
  note={Under review},
  year={2026}
}
```

## 5. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
