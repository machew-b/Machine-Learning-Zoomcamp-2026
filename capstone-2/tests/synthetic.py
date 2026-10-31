"""Synthetic data in the exact format of diabetic_data.csv (for tests only, not real patient data).

The generator reproduces what matters for the code paths: the 50 column names (with hyphens), the
text values ('?' for missing, 'None' for a lab test that was not performed, 'NO' / '>30' / '<30' as the target),
repeated patients, expired / hospice discharges, and a target that depends on the inputs (plus a patient effect),
so that models can learn something and leakage across a patient's encounters can be demonstrated.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

def _file_name(column: str) -> str:
    """Canonical (underscore) name -> name used in the real csv header."""
    if column in ("glyburide_metformin", "glipizide_metformin", "glimepiride_pioglitazone",
                  "metformin_rosiglitazone", "metformin_pioglitazone"):
        return column.replace("_", "-")
    return column


def make_raw_frame(n_rows: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A frame of strings with the real file's header (hyphenated drug names)."""
    rng = np.random.default_rng(seed)
    n_patients = max(10, int(n_rows * 0.7))
    patient_index = rng.integers(0, n_patients, n_rows)
    patient_effect = rng.normal(0.0, 0.5, n_patients)[patient_index]

    def pick(values, probabilities, size=n_rows):
        probabilities = np.asarray(probabilities, dtype=float)
        return rng.choice(values, size=size, p=probabilities / probabilities.sum())

    age = pick(common.AGE_LABELS, [1, 2, 4, 8, 14, 19, 24, 22, 10, 2])
    age_mid = np.array([common.AGE_MID[a] for a in age])
    time_in_hospital = np.clip(rng.poisson(3.3, n_rows) + 1, 1, 14)
    number_inpatient = np.minimum(rng.poisson(0.6, n_rows) + (rng.random(n_rows) < 0.05) * rng.integers(2, 12, n_rows), 21)
    number_emergency = np.minimum(rng.poisson(0.2, n_rows), 30)
    number_outpatient = np.minimum(rng.poisson(0.4, n_rows), 30)
    num_medications = np.clip(np.round(rng.normal(16, 8, n_rows)), 1, 80).astype(int)
    number_diagnoses = np.clip(rng.poisson(6.5, n_rows) + 1, 1, 16)

    insulin = pick(common.MED_STATUS, [46, 30, 12, 12])
    metformin = pick(common.MED_STATUS, [80, 18, 1, 1])
    meds = {}
    for drug in common.MED_COLUMNS:
        if drug == "insulin":
            meds[drug] = insulin
        elif drug == "metformin":
            meds[drug] = metformin
        elif drug in ("examide", "citoglipton", "troglitazone"):
            meds[drug] = np.array(["No"] * n_rows)
        elif drug in ("glipizide", "glyburide", "glimepiride", "pioglitazone", "rosiglitazone"):
            meds[drug] = pick(common.MED_STATUS, [85, 12, 1.5, 1.5])
        else:
            meds[drug] = pick(common.MED_STATUS, [99.4, 0.5, 0.05, 0.05])
    med_frame = pd.DataFrame(meds)
    any_changed = med_frame.isin(["Up", "Down"]).any(axis=1).to_numpy()
    any_drug = (med_frame != "No").any(axis=1).to_numpy()

    codes = ["250.83", "250.01", "428", "414", "786", "715", "491", "585", "V57", "E909", "276", "038", "434",
             "401", "599", "486", "578", "820", "162", "590"]
    diag_1 = pick(codes, np.ones(len(codes)))
    diag_2 = np.where(rng.random(n_rows) < 0.03, "?", pick(codes, np.ones(len(codes))))
    diag_3 = np.where(rng.random(n_rows) < 0.12, "?", pick(codes, np.ones(len(codes))))
    glu = pick(common.GLU_VALUES, [94, 2.5, 1.5, 2])
    a1c = pick(common.A1C_VALUES, [83, 5, 8, 4])

    # three-class target: softmax of linear scores plus the patient effect
    base = 0.30 * number_inpatient + 0.10 * number_emergency + 0.02 * (age_mid - 65) / 10 * 5
    score_late = -0.70 + 0.9 * base + 0.03 * time_in_hospital + 0.4 * patient_effect
    score_early = -2.35 + 1.3 * base + 0.04 * time_in_hospital + 0.25 * (a1c == ">8") + 0.5 * patient_effect
    scores = np.column_stack([np.zeros(n_rows), score_late, score_early])
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    draws = rng.random(n_rows)
    target = (draws[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)

    discharge = pick([1, 3, 6, 2, 22, 5, 4, 11, 13, 14, 18, 25, 19, 20], [60, 14, 12, 4, 6, 3, 1, 2, 0.5, 0.5, 4, 1, 0.1, 0.1])

    frame = pd.DataFrame(
        {
            "encounter_id": rng.permutation(n_rows) + 2278392,
            "patient_nbr": patient_index + 8222157,
            "race": pick(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], [75, 19, 2, 0.6, 1.5, 2.2]),
            "gender": pick(["Female", "Male", "Unknown/Invalid"], [53.8, 46.2, 0.01]),
            "age": age,
            "weight": pick(["?", "[75-100)", "[50-75)", "[100-125)"], [96.9, 1.4, 0.9, 0.8]),
            "admission_type_id": pick([1, 2, 3, 5, 6, 7, 8], [53, 18, 19, 5, 5, 0.02, 0.3]),
            "discharge_disposition_id": discharge,
            "admission_source_id": pick([7, 1, 4, 6, 2, 5, 3, 9, 17, 20], [56, 29, 3, 2, 1, 1, 1, 0.5, 6, 0.5]),
            "time_in_hospital": time_in_hospital,
            "payer_code": pick(["?", "MC", "MD", "HM", "UN", "BC", "SP", "CP"], [40, 32, 3.5, 6, 2.5, 4.5, 5, 2.5]),
            "medical_specialty": pick(
                ["?", "InternalMedicine", "Cardiology", "Family/GeneralPractice", "Surgery-General",
                 "Emergency/Trauma", "Orthopedics", "Nephrology"], [49, 14, 5, 7, 3, 7.5, 1.4, 1.6]),
            "num_lab_procedures": np.clip(np.round(rng.normal(43, 19, n_rows)), 1, 120).astype(int),
            "num_procedures": np.minimum(rng.poisson(1.3, n_rows), 6),
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": glu,
            "A1Cresult": a1c,
        }
    )
    for drug in common.MED_COLUMNS:
        frame[drug] = meds[drug]
    frame["change"] = np.where(any_changed, "Ch", "No")
    frame["diabetesMed"] = np.where(any_drug, "Yes", "No")
    frame["readmitted"] = np.array(common.CLASS_LABELS)[target]
    frame = frame.rename(columns={column: _file_name(column) for column in frame.columns})
    return frame


def to_csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def write_csv(path, n_rows: int = 3000, seed: int = 0) -> pd.DataFrame:
    frame = make_raw_frame(n_rows, seed)
    Path(path).write_text(to_csv_text(frame), encoding="latin-1", newline="")
    return frame


def make_zip_bytes(csv_text: str) -> bytes:
    """A zip shaped like the UCI download: the data file plus IDS_mapping.csv."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(common.CSV_NAME, csv_text)
        archive.writestr("IDS_mapping.csv", "admission_type_id,description\n1,Emergency\n")
    return buffer.getvalue()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_diabetic_data.csv")
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    write_csv(out, rows)
    print(f"wrote {out.name} with {rows} rows")
