# Code map: paper terms → package names

| Term | Meaning in this package |
|------|-------------------------|
| **InterLake-QC / D2** | Same-pass inter-lake consistency detector (`core.run_d2_final`, API `detect`) |
| **B-temp** | Per-lake temporal baseline (`baselines.run_btemp`) |
| **strict** | Strict SWOT QC (`quality_f=0`, `partial_f=0`, `dark_frac<0.5`) |
| **q01** | Keep observations with `quality_f ∈ {0,1}` |
| **S1 / S2** | Synthetic injection / known-truth experiments (`synthetic_engine`) |
| **R4b** | Full synthetic + low-water checks → `scripts/31_synthetic_full.py`, outputs `r4b_*` |
| **R5** | Mahakam final D2 + ablation → `scripts/20_run_mahakam.py`, outputs `r5_*` |
| **R6** | OAT / LOYO / by-pass sensitivity → `scripts/22_sensitivity_mahakam.py`, outputs `r6_*` |
| **R7b** | Yahara gage validation → `scripts/21_run_yahara.py`, outputs `r7b_*` |

## Engine module renames (F6a-2)

| Old | New |
|-----|-----|
| `d2_interlake.py` | `core.py` |
| `d2b_tri.py` | `triangulation.py` |
| `d2b_common.py` | `common.py` |
| `d2b_synth.py` | `synthetic_engine.py` |
| `r5_ablation.py` | `validation_mahakam.py` |
| `r6_sensitivity.py` | `sensitivity.py` |
| `r7_yahara.py` | `validation_yahara.py` |

Public wrappers: `pairing`, `reference`, `detect`, `attribution`, `baselines`, `validate`, `synthetic`.
