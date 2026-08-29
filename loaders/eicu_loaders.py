"""Batched Neo4j ingestion helpers for credentialed eICU tables.

These functions load the source schema used by the benchmark. Medication-name
normalization and the fixed 147-candidate construction are separate stages;
their generated tables are intentionally not redistributed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from neo4j import Driver

LOGGER = logging.getLogger(__name__)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], caller: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{caller}: missing columns: {missing}")


def _run_frame_in_batches(
    driver: Driver,
    cypher: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    database: str,
    batch_size: int,
) -> int:
    """Send a DataFrame to an ``UNWIND $rows`` query without one large list."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    total = len(frame)
    with driver.session(database=database) as session:
        for start in range(0, total, batch_size):
            batch = frame.iloc[start : start + batch_size][list(columns)].copy()
            batch = batch.astype(object).where(pd.notna(batch), None)
            rows = batch.to_dict(orient="records")
            session.run(cypher, rows=rows).consume()
    return total


def load_eicu_patient(
    patient_df: pd.DataFrame,
    driver: Driver,
    database: str = "neo4j",
    batch_size: int = 5_000,
) -> int:
    """Load patient and visit nodes plus ``HAS_VISIT`` edges."""

    columns = [
        "uniquepid",
        "patientunitstayid",
        "gender",
        "age",
        "ethnicity",
        "unittype",
        "unitadmittime24",
        "unitdischargetime24",
    ]
    _require_columns(patient_df, columns, "load_eicu_patient")
    cypher = """
    UNWIND $rows AS row
    MERGE (p:Patient {patient_id: row.uniquepid})
      SET p.gender = row.gender,
          p.age = row.age,
          p.ethnicity = row.ethnicity
    MERGE (v:Visit {visit_id: row.patientunitstayid})
      SET v.unittype = row.unittype,
          v.unitadmittime24 = row.unitadmittime24,
          v.unitdischargetime24 = row.unitdischargetime24
    MERGE (p)-[:HAS_VISIT]->(v)
    """
    total = _run_frame_in_batches(
        driver,
        cypher,
        patient_df,
        columns,
        database=database,
        batch_size=batch_size,
    )
    LOGGER.info("Loaded %d patient/visit rows", total)
    return total


def load_eicu_diagnosis(
    diagnosis_df: pd.DataFrame,
    driver: Driver,
    database: str = "neo4j",
    batch_size: int = 5_000,
) -> int:
    """Load diagnosis-event nodes and ``HAS_DIAGNOSIS_EVENT`` edges."""

    columns = [
        "diagnosisid",
        "patientunitstayid",
        "diagnosisstring",
        "icd9code",
        "diagnosisoffset",
        "diagnosispriority",
        "activeupondischarge",
    ]
    _require_columns(diagnosis_df, columns, "load_eicu_diagnosis")
    cypher = """
    UNWIND $rows AS row
    MERGE (v:Visit {visit_id: row.patientunitstayid})
    MERGE (d:DiagnosisEvent {diagnosis_id: row.diagnosisid})
      SET d.diagnosisstring = row.diagnosisstring,
          d.icd9code = row.icd9code,
          d.diagnosisoffset = row.diagnosisoffset,
          d.diagnosispriority = row.diagnosispriority,
          d.activeupondischarge = row.activeupondischarge
    MERGE (v)-[:HAS_DIAGNOSIS_EVENT]->(d)
    """
    total = _run_frame_in_batches(
        driver,
        cypher,
        diagnosis_df,
        columns,
        database=database,
        batch_size=batch_size,
    )
    LOGGER.info("Loaded %d diagnosis rows", total)
    return total


def _clean_gtc(series: pd.Series) -> pd.Series:
    """Normalize integral GTC codes while preserving valid code zero."""

    numeric = pd.to_numeric(series, errors="coerce")
    integral = numeric.notna() & np.isfinite(numeric) & (numeric == np.floor(numeric))
    output = pd.Series(pd.NA, index=series.index, dtype="string")
    output.loc[integral] = numeric.loc[integral].astype(np.int64).astype(str)
    return output


def load_eicu_medication(
    medication_df: pd.DataFrame,
    driver: Driver,
    database: str = "neo4j",
    batch_size: int = 5_000,
) -> int:
    """Load drug events, visit edges, and source GTC concept edges.

    RxNorm normalization is performed in the separate credentialed mapping
    stage. A GTC value of zero is retained to match the frozen benchmark
    vocabulary. The release treats it as a dataset-specific source category;
    it does not infer a clinical meaning for that code.
    """

    source_columns = [
        "medicationid",
        "patientunitstayid",
        "drugname",
        "dosage",
        "routeadmin",
        "frequency",
        "drugstartoffset",
        "drugstopoffset",
        "gtc",
    ]
    _require_columns(medication_df, source_columns, "load_eicu_medication")
    clean = medication_df.copy()
    clean["gtc_clean"] = _clean_gtc(clean["gtc"])
    clean = clean.dropna(subset=["gtc_clean"])
    columns = [*source_columns[:-1], "gtc_clean"]
    cypher = """
    UNWIND $rows AS row
    MERGE (v:Visit {visit_id: row.patientunitstayid})
    MERGE (m:DrugExposureEvent {medication_id: row.medicationid})
      SET m.drugname = row.drugname,
          m.dosage = row.dosage,
          m.routeadmin = row.routeadmin,
          m.frequency = row.frequency,
          m.drugstartoffset = row.drugstartoffset,
          m.drugstopoffset = row.drugstopoffset
    MERGE (m)-[:IN_VISIT]->(v)
    MERGE (c:Concept {concept_id: 'GTC:' + row.gtc_clean})
    MERGE (m)-[:DRUG_OF_CONCEPT]->(c)
    """
    total = _run_frame_in_batches(
        driver,
        cypher,
        clean,
        columns,
        database=database,
        batch_size=batch_size,
    )
    LOGGER.info("Loaded %d medication rows with integral GTC codes", total)
    return total


def load_eicu_lab(
    lab_df: pd.DataFrame,
    driver: Driver,
    database: str = "neo4j",
    batch_size: int = 5_000,
) -> int:
    """Load laboratory events for descriptive auditing, not model features."""

    columns = [
        "labid",
        "patientunitstayid",
        "labresultoffset",
        "labname",
        "labresult",
        "labresulttext",
        "labmeasurenamesystem",
        "labmeasurenameinterface",
    ]
    _require_columns(lab_df, columns, "load_eicu_lab")
    cypher = """
    UNWIND $rows AS row
    MERGE (v:Visit {visit_id: row.patientunitstayid})
    MERGE (m:MeasurementEvent {lab_id: row.labid})
      SET m.labname = row.labname,
          m.labresult = row.labresult,
          m.labresulttext = row.labresulttext,
          m.labresultoffset = row.labresultoffset,
          m.labmeasurename_system = row.labmeasurenamesystem,
          m.labmeasurename_interface = row.labmeasurenameinterface
    MERGE (m)-[:IN_VISIT]->(v)
    """
    total = _run_frame_in_batches(
        driver,
        cypher,
        lab_df,
        columns,
        database=database,
        batch_size=batch_size,
    )
    LOGGER.info("Loaded %d laboratory rows for descriptive auditing", total)
    return total
