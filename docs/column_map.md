# Column and value map (legacy → English)

Golden reference CSVs keep legacy (Indonesian / audit) names. Package outputs and configs use English names. Regression tests convert goldens via `interlake_qc.schema`.

## Columns

| Legacy | English | Unit / meaning |
|--------|---------|----------------|
| `danau` | `lake` | lake name |
| `danau_a` / `danau_b` / `danau_tri` | `lake_a` / `lake_b` / `lake_tri` | pair endpoints / triangulation lake |
| `sumber_akhir` | `final_source` | attributed guilty lake or `ambiguous` |
| `sumber_tri` | `triangulation_source` | triangulation-only source |
| `sumber_temporal` | `temporal_source` | temporal-only source |
| `cara_atribusi` | `attribution_method` | `triangulation` / `temporal` / `ambiguous` |
| `bermasalah` | `anomalous` | truth label \|e_dwse_c\| > T |
| `himpunan` | `subset` | evaluation subset id |
| `peran` | `role` | `primary` / `sensitivity` / `small_object` |
| `tanggal` | `date` | UTC calendar date |
| `k_dipakai` | `k_used` | k threshold applied to the pass |
| `ketat` | `strict` | strict QC filter (config + method name) |
| `pasangan` | `pair` | pair name |
| `metode` | `method` | detection/filter method |
| `n_bermasalah` | `n_anomalous` | count of anomalous passes |
| `n_ditandai` | `n_flagged` | count of flagged/dropped passes |
| `n_pasangan` | `n_pairs` | ATL13–SWOT coincidence count |
| `presisi` (+ `_lo`/`_hi`) | `precision` (+ `_lo`/`_hi`) | Wilson CI |
| `frac_dipertahankan` | `frac_retained` | 1 − flagged fraction |
| `benar` / `salah` / `ambigu_attr` | `correct` / `incorrect` / `ambiguous_attr` | attribution scores |
| `sudah_ditandai_flag` | `already_quality_flagged` | SWOT quality already bad |

## Categorical values

| Legacy | English |
|--------|---------|
| `triangulasi` | `triangulation` |
| `temporal` | `temporal` |
| `ambigu` | `ambiguous` |
| `tak bisa diatribusi` | `unattributed` |
| `utama` | `primary` |
| `sensitivitas` | `sensitivity` |
| `objek_kecil` | `small_object` |
| `utama_ice0_q01` | `primary_ice0_q01` |
| `ketat` (method) | `strict` |
| `sama` / `tambah` / `hilang` / `tetap_bersih` | `same` / `added` / `lost` / `stays_clean` |
| `V2_ketat` | `V2_strict` |
| `n_jempang_ketat` | `n_jempang_strict` |
| `utama_POSTHOC` | `primary_POSTHOC` |

Output **file names** (`r5_*`, `r6_*`, `r7b_*`, `r4b_*`) are unchanged; see `docs/code_map.md` for prefixes.
