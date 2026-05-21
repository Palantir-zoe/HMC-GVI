from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, add_intercept, standardize_columns


def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    return df


def load_pima_dataset() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "data" / "pima.csv"
    df = _drop_index_column(pd.read_csv(path))
    y = (df["type"] == "Yes").astype(int).to_numpy(dtype=float)
    x = df.drop(columns=["type"]).to_numpy(dtype=float)
    return y, add_intercept(standardize_columns(x))


def load_german_credit_dataset() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "data" / "german_credit.csv"
    df = pd.read_csv(path)
    y = (df["Creditability"] == 1).astype(int).to_numpy(dtype=float)
    x = df.drop(columns=["Creditability"]).to_numpy(dtype=float)
    return y, add_intercept(standardize_columns(x))


def load_polypharm_dataset() -> dict[str, np.ndarray]:
    path = ROOT / "data" / "polypharmacy.csv"
    df = _drop_index_column(pd.read_csv(path))

    n_subjects = int(df["id"].nunique())
    repeats = df.groupby("id").size().iloc[0]

    design = np.ones((len(df), 8), dtype=float)
    design[:, 1] = (df["gender"] == "Male").astype(float)
    design[:, 2] = (df["race"] != "White").astype(float)
    design[:, 3] = np.log(df["age"].to_numpy(dtype=float) / 10.0)
    design[:, 4] = (df["mhv4"] == "1-5").astype(float)
    design[:, 5] = (df["mhv4"] == "6-14").astype(float)
    design[:, 6] = (df["mhv4"] == "> 14").astype(float)
    design[:, 7] = (df["inptmhv3"] != "0").astype(float)
    response = (df["polypharmacy"] == "Yes").astype(float).to_numpy()

    return {
        "n_subjects": n_subjects,
        "repeats": int(repeats),
        "x": design,
        "y": response,
    }

