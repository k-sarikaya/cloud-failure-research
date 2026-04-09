# Dynamic Risk Mitigation in Edge Computing: A Queuing-Theoretic Framework for Reliability-Important Cascading Failure Prevention

This repository is synchronized to the round-1 revision package for the Reliability Engineering and System Safety submission.

The authoritative reproduction path for the revised manuscript now lives under `code/` and `artifacts/`.

Important note:
- `fc_dqn_archive/` contains `simulation_code.py`, `dqn_weights.pt`, `results_analysis.ipynb`, the archived `results/` directory, and the pre-revision `trained_agent_*.json` checkpoints.
- The revised manuscript, supplementary materials, and response letter are aligned to the `code/` and `artifacts/` snapshot added here. If you are reproducing the revised submission, ignore `fc_dqn_archive/` unless you specifically need historical traceability.

## Packaged headline benchmark

Canonical paired benchmark: BA-100, 40 frozen paired scenarios, laptop-class CPU replay.

| Method | Mean SST (h) |
| :-- | --: |
| No Defence | 4.50 |
| Heuristic Pruning (Graceful Drain) | 5.49 |
| +25% Backup | 7.93 |
| +50% Backup | 10.82 |
| FC-DQN | 0.59 |
| GAT-SAC (Ours) | 14.03 |

Rounded to the manuscript's 2-decimal convention; `artifacts/verified_metrics.json` retains the full-precision values used for reproducibility.

## Canonical calibration artifacts

The revision package now uses a single proxy-based calibration pipeline for Figure 3, built from 334 retained rows drawn from a 358-incident source dataset.
The authoritative inputs and outputs are:

Use only `code/`, `artifacts/`, and `data/processed/canonical_calibration/` for revised-package reproduction.

- `data/processed/processed_with_load_proxies.csv`
- `data/processed/canonical_calibration/calibration_report.txt`
- `data/processed/canonical_calibration/calibration_summary.json`
- `data/processed/canonical_calibration/calibration_loocv.csv`
- `data/processed/canonical_calibration/fig3_calibration.pdf`
- `data/processed/canonical_calibration/azure_vm_crosscheck_binned.csv`
- `data/processed/canonical_calibration/azure_vm_crosscheck_summary.json`
- `data/processed/canonical_calibration/cloud_uptime_archive_crosscheck.csv`
- `data/processed/canonical_calibration/cloud_uptime_archive_crosscheck_summary.json`

Important note:
- `data/processed/final_failure_logs.csv` is not the canonical Figure 3 source because its `cpu_load` and `memory_load` columns are placeholder load values from the earlier preprocessing pipeline.
- The canonical proxy-based fit yields a positive but weakly identified slope centered near `alpha ~= 0.9` (`alpha = 0.894`, bootstrap median `0.851`, 95% CI `[-0.480, 2.253]`, `R^2 = 0.0055`).
- We therefore use `alpha = 0.9` as a pragmatic stressed-system reference value rather than a tightly identified empirical constant, while the packaged BA-100 benchmark remains fixed at `alpha = 1.2`.
- The main manuscript claims are intentionally bounded to the canonical BA-100 benchmark; topology and size-transfer checks remain supplementary descriptive probes.

To regenerate the calibration artifacts:

```bash
python src/analysis/build_canonical_calibration.py
```

To regenerate the Azure telemetry cross-check summary (requires local Azure CPU shard files):

```bash
python src/analysis/build_azure_vm_crosscheck_summary.py --help
```

To regenerate the Cloud Uptime Archive cross-check (requires downloading selected CUA CSV traces from Zenodo):

```bash
python src/analysis/build_cloud_uptime_archive_crosscheck.py --help
```

## Supplementary robustness checks bundled in the repo

- Fixed-policy hazard-slope replay over `alpha in {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}` on the same frozen BA-100 40-scenario bank.
- Fixed-policy BA size transfer (`n = 100, 150, 200`) and WS/ER/5G-MEC topology transfer.
- Illustrative OVHcloud Strasbourg-inspired replay.

For the hazard-slope replay, the qualitative ranking does not reverse across the tested range. The mean GAT-SAC advantage over No Defence grows from `2.558 h` at `alpha = 0.5` to `14.171 h` at `alpha = 1.5`.


## Manuscript figure source map

The revised manuscript figures are now backed by repo-visible source scripts (compiled figure numbers are listed below; some artifact filenames retain their historical numbering):

- Figure 1 (`fig1_framework.pdf`) -> `code/scripts/generate_submission_overview_figures.py`
- Figure 2 (`fig2_cascade.pdf`) -> `code/scripts/generate_submission_overview_figures.py`
- Figure 3 (`fig3_calibration.pdf`) -> `src/analysis/build_canonical_calibration.py`
- Figure 4 (compiled numbering; file `fig5_sst_violin.pdf`) -> `code/scripts/generate_figures.py`
- Figure 5 (compiled numbering; file `fig8_sensitivity.pdf`) -> `code/scripts/generate_figures.py`

To regenerate the manuscript-compatible figure bundle inside the repo:

```bash
python code/scripts/generate_submission_overview_figures.py
python code/scripts/generate_figures.py
```

These commands write the manuscript-compatible outputs under `main_text/figures/`. Figure 3 remains the canonical calibration output generated by `src/analysis/build_canonical_calibration.py` and stored under `data/processed/canonical_calibration/`.

## Repository layout

- `code/` - synchronized revision code used by the packaged benchmark
- `artifacts/` - synchronized benchmark outputs, frozen scenario bank, and reproducibility manifest
- `incident_data.csv` - incident dataset used in the calibration workflow
- `fc_dqn_archive/` - archival pre-revision FC-DQN code, weights, notebooks, checkpoints, and logs retained only for historical traceability

## Installation

```bash
pip install -r requirements.txt
```

## Canonical eval-only replay

```bash
python code/training/train_sac.py --eval-only \
  --weights code/weights/gat_sac_v8_retrained_seed42.pt \
  --scenario-bank artifacts/scenario_bank_ba_n100_k40_seed42.npz \
  --eval-scenarios 40 \
  --device cpu
```

## Hazard-slope alpha sweep

```bash
python code/scripts/run_alpha_sweep_eval.py \
  --weights code/weights/gat_sac_v8_retrained_seed42.pt \
  --scenario-bank artifacts/scenario_bank_ba_n100_k40_seed42.npz \
  --eval-scenarios 40 \
  --alphas 0.5,0.7,0.9,1.1,1.3,1.5 \
  --out-tag alpha_sweep_ba_n100_k40_seed42
```

## Fixed-policy transfer checks

```bash
python code/scripts/run_topology_scalability.py \
  --topos ba \
  --node-counts 100,150,200 \
  --eval-scenarios 3 \
  --shared-weights code/weights/gat_sac_v8_retrained_seed42.pt \
  --out artifacts/topology_scalability_ba_fixed_policy.csv

python code/scripts/run_topology_scalability.py \
  --topos ws,er,5g_mec \
  --node-counts 100 \
  --eval-scenarios 5 \
  --shared-weights code/weights/gat_sac_v8_retrained_seed42.pt \
  --out artifacts/topology_transfer_topologies_k5.csv
```

## OVHcloud replay

```bash
python code/scripts/run_ovhcloud_replay.py
```

## Key synchronized artifacts

- `artifacts/paired_eval_results.csv`
- `artifacts/paired_eval_summary.csv`
- `artifacts/pairwise_stats.csv`
- `artifacts/alpha_sweep_ba_n100_k40_seed42_results.csv`
- `artifacts/alpha_sweep_ba_n100_k40_seed42_summary.csv`
- `artifacts/alpha_sweep_ba_n100_k40_seed42_meta.json`
- `artifacts/scenario_bank_ba_n100_k40_seed42.npz`
- `artifacts/reproducibility_manifest.txt`
- `artifacts/topology_scalability_ba_fixed_policy.csv`
- `artifacts/topology_transfer_topologies_k5.csv`
- `artifacts/ovhcloud_replay_results.csv`
- `artifacts/ovhcloud_replay_summary.json`

## Citation

If you use this synchronized revision snapshot, please cite the accompanying article metadata in `CITATION.cff`.
