"""English column / value schema and maps from legacy Indonesian names.

Golden reference CSVs keep legacy names; regression tests convert via these maps.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# Legacy (Indonesian / audit) → English output/config names
COLUMN_MAP: dict[str, str] = {
    "danau": "lake",
    "danau_a": "lake_a",
    "danau_b": "lake_b",
    "danau_tri": "lake_tri",
    "sumber": "source",
    "sumber_akhir": "final_source",
    "sumber_tri": "triangulation_source",
    "sumber_temporal": "temporal_source",
    "cara_atribusi": "attribution_method",
    "bermasalah": "anomalous",
    "himpunan": "subset",
    "peran": "role",
    "tanggal": "date",
    "k_dipakai": "k_used",
    "sudah_ditandai_flag": "already_quality_flagged",
    "ketat": "strict",
    "pasangan": "pair",
    "n_pasangan": "n_pairs",
    "n_pair": "n_pairs",
    "metode": "method",
    "n_bermasalah": "n_anomalous",
    "n_ditandai": "n_flagged",
    "n_kept": "n_kept",

    "presisi": "precision",
    "presisi_lo": "precision_lo",
    "presisi_hi": "precision_hi",
    "frac_dipertahankan": "frac_retained",
    "benar": "correct",
    "salah": "incorrect",
    "ambigu_attr": "ambiguous_attr",
    "frac_benar": "frac_correct",
    "frac_salah": "frac_incorrect",
    "frac_ambigu": "frac_ambiguous",
    "guilty_lake": "guilty_lake",
    "fase": "phase",
    "wse_a": "wse_a",
    "wse_b": "wse_b",
    "q_a": "q_a",
    "q_b": "q_b",
}

VALUE_MAP: dict[str, str] = {
    "triangulasi": "triangulation",
    "temporal": "temporal",
    "ambigu": "ambiguous",
    "tak bisa diatribusi": "unattributed",
    "utama": "primary",
    "sensitivitas": "sensitivity",
    "objek_kecil": "small_object",
    "utama_v0": "primary_v0",
    "sekunder_v0": "secondary_v0",
    "utama_ice0_q01": "primary_ice0_q01",
    "q01_noice": "q01_noice",
    "ketat": "strict",
    "sama": "same",
    "tetap_bersih": "stays_clean",
    "tambah": "added",
    "hilang": "lost",
    "V2_ketat": "V2_strict",
    "n_jempang_ketat": "n_jempang_strict",
    "utama_POSTHOC": "primary_POSTHOC",
}

# Columns whose cell values should be remapped
VALUE_COLUMNS = {
    "final_source",
    "triangulation_source",
    "temporal_source",
    "source",
    "attribution_method",
    "cara_atribusi",
    "sumber_akhir",
    "sumber_tri",
    "sumber_temporal",
    "sumber",
    "role",
    "peran",
    "subset",
    "himpunan",
    "method",
    "metode",
    "vs_default",
    "variant",
    "metric",
    "subset_POSTHOC",
}


def rename_columns(df: pd.DataFrame, reverse: bool = False) -> pd.DataFrame:
    mapping = {v: k for k, v in COLUMN_MAP.items()} if reverse else COLUMN_MAP
    out = df.copy()
    rename = {c: mapping[c] for c in out.columns if c in mapping and mapping[c] != c}
    return out.rename(columns=rename)


def remap_values(df: pd.DataFrame, reverse: bool = False) -> pd.DataFrame:
    mapping = {v: k for k, v in VALUE_MAP.items()} if reverse else VALUE_MAP
    out = df.copy()
    for c in out.columns:
        if c not in VALUE_COLUMNS and c not in COLUMN_MAP:
            # also remap if column is a known value-bearing English name
            if c not in {
                "final_source",
                "triangulation_source",
                "temporal_source",
                "attribution_method",
                "role",
                "subset",
                "method",
                "source",
            }:
                continue
        if c not in out.columns:
            continue
        if out[c].dtype == object or str(out[c].dtype) in {"string", "str"} or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda x: mapping.get(str(x), x) if pd.notna(x) else x)
    return out


def golden_to_english(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a legacy golden CSV to the English schema for comparison."""
    return remap_values(rename_columns(df, reverse=False), reverse=False)


def cfg_get_pair(cfg: dict) -> dict[str, Any]:
    """Normalize pair block to lake_a/lake_b/lake_tri (accept legacy keys)."""
    p = dict(cfg.get("pair") or {})
    if "lake_a" not in p and "danau_a" in p:
        p["lake_a"] = p["danau_a"]
    if "lake_b" not in p and "danau_b" in p:
        p["lake_b"] = p["danau_b"]
    if "lake_tri" not in p and "danau_tri" in p:
        p["lake_tri"] = p["danau_tri"]
    # keep dual keys for transitional code
    if "lake_a" in p:
        p["danau_a"] = p["lake_a"]
    if "lake_b" in p:
        p["danau_b"] = p["lake_b"]
    if "lake_tri" in p:
        p["danau_tri"] = p["lake_tri"]
    return p


def cfg_get_strict(cfg: dict) -> dict:
    return cfg.get("strict") or cfg.get("ketat") or {}


def normalize_swot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input SWOT-like tables to English column names."""
    out = rename_columns(df)
    if "lake" not in out.columns and "danau" in df.columns:
        out["lake"] = df["danau"]
    return out
