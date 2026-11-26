# Data

## Processed inputs (committed; offline reproduction)

| File | Source | Access / DOI | License note |
|------|--------|--------------|--------------|
| `processed/mahakam_swot_lakesp.csv` | SWOT LakeSP (version D) via Hydrocron, as used in audit `r2_swot_with_pass.csv` | Access date: see audit Hydrocron logs | See NASA Earthdata data policy |
| `processed/mahakam_atl13_daily_v2.csv` | ICESat-2 ATL13 v007, Luo & Song V2 daily QC (NMAD factor 1.4826) from audit `r1_atl13_daily_V2.csv` | via NASA Earthdata | See NASA Earthdata data policy |
| `processed/mahakam_atl13_interlake.csv` | ATL13 inter-lake differences from audit `r2_atl13_interlake.csv` | derived | See NASA Earthdata data policy |
| `processed/mahakam_pairs.csv` | Same-pass Semayang−Melintang pairs (audit `r4_pairs_semayang_melintang.csv`) | derived | — |
| `processed/mahakam_dwse_phase.csv` | Phase labels (audit `r3_dwse_with_phase.csv`) | derived | — |
| `processed/yahara_usgs_subset.csv` | USGS SWOT–gage paired lakes (Harlan et al. 2026), four Yahara `lake_id`s only | doi:10.5066/P1FE9W9E | CC0 |

`data/raw/` is not committed (online fetch only).

## Columns (processed)

- **mahakam_swot_lakesp**: `lake_id`, `time`, `danau`, `wse`, `quality_f`, `partial_f`, `dark_frac`, `pass`, `cycle`, …
- **mahakam_atl13_daily_v2**: `danau`, `tanggal`, `n_seg`, `mean`, `std`, `n_seg_before_nmad`
- **yahara_usgs_subset**: `gage_agency`, `gage_id`, `lake_id`, `SWOT_time`, `gage_time_UTC`, `gage_wse_EGM2008`, `wse`, `quality_f`, `partial_f`, `dark_frac`, `ice_clim_f`, `cycle_id`, `pass_id`, `SWOT_data_version`, …
