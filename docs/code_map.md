# Code map: internal terms → public names

| Internal / paper shorthand | Public name in this package |
|----------------------------|-----------------------------|
| D2 / InterLake-QC detector | **InterLake-QC** (`core.run_d2_final`, API: `detect`) |
| B-temp | Per-lake temporal baseline (`baselines.run_btemp`) |
| f(H), σ(H) reference curve | `reference.build_reference_bins` |
| Same-pass pairing | `pairing.prepare_pair_table` / `pairing.build_pair_table` |
| Triangulation (Jempang / Mendota) | `triangulation` / API `attribution` |
| Ketat / q∈{0,1} | `baselines.mask_ketat` / `mask_q01` |
| S1 / S2 synthetic | `synthetic_engine` / API `synthetic` |
| R4b | `scripts/31_synthetic_full.py`, `configs/synthetic.yaml` → `outputs/synthetic/` |
| R5 Mahakam | `scripts/20_run_mahakam.py` → `outputs/mahakam/r5_*` |
| R6 sensitivity | `scripts/22_sensitivity_mahakam.py` → `outputs/mahakam/r6_*` |
| R7b Yahara | `scripts/21_run_yahara.py` → `outputs/yahara/r7b_*` |
| Ablation V0–V6 | `validation_mahakam` |
| Yahara metrics / RMSE | `validation_yahara` |
| Low-water ATL13 checks | `lowwater` |

## Renamed engine modules (Fase 6a-2)

| Old audit-style name | New name |
|----------------------|----------|
| `d2_interlake.py` | `core.py` |
| `d2b_tri.py` | `triangulation.py` |
| `d2b_common.py` | `common.py` |
| `d2b_synth.py` | `synthetic_engine.py` |
| `r5_ablation.py` | `validation_mahakam.py` |
| `r6_sensitivity.py` | `sensitivity.py` |
| `r7_yahara.py` | `validation_yahara.py` |
| `synthetic_r4.py` | **removed** (circular R4 synth rejected; not used in manuscript) |

Public wrappers (`pairing`, `reference`, `detect`, `attribution`, `baselines`, `validate`, `synthetic`) remain the preferred import surface. Output CSV column names are unchanged in this phase.
