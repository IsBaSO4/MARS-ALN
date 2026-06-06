#!/usr/bin/env python3
"""Write a clinical xlsx in the canonical schema that ``ALNpCRDataset`` reads.

The dataset expects a sheet named ``已完成`` with a **2-row header** (a group-label
row + the real column-name row) and these canonical columns:

    ID.1                              patient ID (must match imaging folder names)
    HER2(1:positive;0:negative)       biopsy HER2 status (0/1)
    ER(数值)                          biopsy ER  (numeric or 0/1)
    PR(数值)                          biopsy PR  (numeric or 0/1)
    HR(1:positive;0:negative)         hormone-receptor status (0/1; derived if absent)
    Ki67(数值)                        biopsy Ki67 (numeric or 0/1)
    post-LN（1:positive；0:negative）  post-NAT axillary nodal status  ->  label = 1 - post-LN

This tool maps your source columns onto that schema and writes the file. If your
source columns are already named canonically, omit ``--map``.

Usage:
    python scripts/align_clinical.py --input raw.xlsx --output clinical_aligned.xlsx \
        --sheet Sheet1 --header 0 \
        --map "pid=ID.1,HER2_status=HER2(1:positive;0:negative),ER=ER(数值),PR=PR(数值),Ki67=Ki67(数值),ypN=post-LN（1:positive；0:negative）"
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("align_clinical")

COL_ID = "ID.1"
COL_HER2 = "HER2(1:positive;0:negative)"
COL_ER = "ER(数值)"
COL_PR = "PR(数值)"
COL_HR = "HR(1:positive;0:negative)"
COL_KI67 = "Ki67(数值)"
COL_POST_LN = "post-LN（1:positive；0:negative）"
COL_PRE_LN = "pre-LN（1:positive；0:negative）"

CANONICAL_COLUMNS = [COL_ID, COL_HER2, COL_ER, COL_PR, COL_HR, COL_KI67, COL_POST_LN, COL_PRE_LN]
CANONICAL_GROUPS = {
    COL_ID: "id", COL_HER2: "biopsy", COL_ER: "biopsy", COL_PR: "biopsy",
    COL_HR: "biopsy", COL_KI67: "biopsy", COL_POST_LN: "pathology", COL_PRE_LN: "pathology",
}


def _clean_id(v) -> str:
    """Coerce an ID cell to str, stripping only a literal trailing ``.0``."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def _parse_map(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        if "=" in pair:
            src, canon = pair.split("=", 1)
            out[src.strip()] = canon.strip()
    return out


def align(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    if column_map:
        df = df.rename(columns=column_map)
    out = pd.DataFrame()
    for c in CANONICAL_COLUMNS:
        if c in df.columns:
            out[c] = df[c]
    if COL_ID not in out.columns:
        raise ValueError(f"Missing required ID column '{COL_ID}'. Use --map to point at it.")
    out[COL_ID] = out[COL_ID].apply(_clean_id)
    for c in (COL_ER, COL_PR, COL_KI67, COL_HER2, COL_POST_LN, COL_PRE_LN):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Derive HR = OR(ER, PR) if missing
    if COL_HR not in out.columns and COL_ER in out.columns and COL_PR in out.columns:
        er = out[COL_ER].fillna(0)
        pr = out[COL_PR].fillna(0)
        hr = ((er >= 0.5) | (pr >= 0.5)).astype("Int64")
        hr[out[COL_ER].isna() & out[COL_PR].isna()] = pd.NA
        out[COL_HR] = hr
        logger.info("Derived HR = (ER OR PR): %d positives", int((hr == 1).sum()))

    return out[[c for c in CANONICAL_COLUMNS if c in out.columns]]


def write_canonical(df: pd.DataFrame, output_path: str) -> None:
    """Write the 2-row-header layout (group row + name row) on sheet '已完成'."""
    cols = list(df.columns)
    group_row = [CANONICAL_GROUPS.get(c, "") for c in cols]
    data_rows = df.astype(object).where(pd.notna(df), "").values.tolist()
    out = pd.DataFrame([group_row, cols] + data_rows, columns=cols)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="已完成", index=False, header=False)
    logger.info("Wrote %d rows x %d cols -> %s", len(df), len(cols), output_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Align a clinical xlsx to the canonical MARS schema")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sheet", default=0, help="source sheet name or index (default: first)")
    p.add_argument("--header", type=int, default=0, help="header row index in the source")
    p.add_argument("--map", default=None, help='source=canonical pairs, e.g. "pid=ID.1,ypN=post-LN（1:positive；0:negative）"')
    args = p.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.input, sheet_name=sheet, header=args.header)
    aligned = align(df, _parse_map(args.map))
    write_canonical(aligned, args.output)
    if COL_POST_LN in aligned.columns:
        logger.info("post-LN distribution: %s",
                    aligned[COL_POST_LN].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
