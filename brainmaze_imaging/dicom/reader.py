"""
DICOM metadata reader.

Recursively walks a directory tree, reads every DICOM file it finds, groups
files by series (``SeriesInstanceUID``), and returns a
:class:`pandas.DataFrame` where **each row represents one scan / series**.

The tag selection follows the same conventions used by **dcm2niix** so that the
resulting DataFrame is immediately useful for quality-control workflows and for
driving DICOM-to-NIfTI conversions.

Typical usage
-------------
>>> from brainmaze_imaging.dicom import load_dicom_metadata
>>> df = load_dicom_metadata("/path/to/dicom/root")
>>> print(df[["subject_id", "series_description", "number_of_slices"]])
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag mapping
# ---------------------------------------------------------------------------
# Each entry maps a friendly column name to a DICOM tag keyword (pydicom
# attribute name).  All fields are optional – if a tag is absent from a
# particular series the column value will be None / NaN.
#
# Only series-level (scan-level) tags are included here.  Slice-specific
# attributes (SOPInstanceUID, InstanceNumber, ImagePositionPatient,
# SliceLocation) are intentionally omitted because they do not carry
# meaningful information at the series level.
#
# The selection is inspired by dcm2niix's behaviour; see:
#   https://github.com/rordenlab/dcm2niix
# ---------------------------------------------------------------------------

#: Map of output column name -> pydicom keyword (or None for derived columns).
DICOM_TAGS: dict[str, str | None] = {
    # Patient / Subject
    "subject_id":                  "PatientID",
    "subject_name":                "PatientName",
    "subject_birth_date":          "PatientBirthDate",
    "subject_sex":                 "PatientSex",
    "subject_weight_kg":           "PatientWeight",
    "subject_age":                 "PatientAge",

    # Study / Series identification
    "study_instance_uid":          "StudyInstanceUID",
    "series_instance_uid":         "SeriesInstanceUID",
    "study_id":                    "StudyID",
    "series_number":               "SeriesNumber",

    # Dates and times
    "acquisition_date":            "AcquisitionDate",
    "acquisition_time":            "AcquisitionTime",
    "series_date":                 "SeriesDate",
    "study_date":                  "StudyDate",
    "series_time":                 "SeriesTime",

    # Equipment
    "manufacturer":                "Manufacturer",
    "manufacturer_model":          "ManufacturerModelName",
    "station_name":                "StationName",
    "software_versions":           "SoftwareVersions",
    "magnetic_field_strength_T":   "MagneticFieldStrength",

    # Modality / Sequence
    "modality":                    "Modality",
    "image_type":                  "ImageType",
    "series_description":          "SeriesDescription",
    "protocol_name":               "ProtocolName",
    "sequence_name":               "SequenceName",
    "scanning_sequence":           "ScanningSequence",
    "sequence_variant":            "SequenceVariant",
    "scan_options":                "ScanOptions",

    # Geometry / Resolution
    "rows":                        "Rows",
    "columns":                     "Columns",
    "slice_thickness_mm":          "SliceThickness",
    "spacing_between_slices_mm":   "SpacingBetweenSlices",
    "pixel_spacing":               "PixelSpacing",
    "pixel_spacing_row_mm":        None,
    "pixel_spacing_col_mm":        None,
    "reconstruction_diameter_mm":  "ReconstructionDiameter",
    "field_of_view_mm":            None,
    "image_orientation_patient":   "ImageOrientationPatient",
    "number_of_slices":            None,

    # MRI timing / contrast
    "repetition_time_ms":          "RepetitionTime",
    "echo_time_ms":                "EchoTime",
    "inversion_time_ms":           "InversionTime",
    "flip_angle_deg":              "FlipAngle",
    "echo_train_length":           "EchoTrainLength",
    "echo_numbers":                "EchoNumbers",

    # MRI acquisition parameters
    "nex":                         "NumberOfAverages",
    "percent_sampling":            "PercentSampling",
    "percent_phase_fov":           "PercentPhaseFieldOfView",
    "pixel_bandwidth_hz":          "PixelBandwidth",
    "phase_encoding_direction":    "InPlanePhaseEncodingDirection",
    "number_of_phase_encoding_steps": "NumberOfPhaseEncodingSteps",
    "phase_encoding_steps_out":    "NumberOfPhaseEncodingStepsOutOfPlane",

    # MRI parallel imaging / SNR
    "parallel_reduction_factor":   "ParallelReductionFactorInPlane",
    "parallel_technique":          "ParallelAcquisitionTechnique",

    # MRI coils / RF
    "transmit_coil":               "TransmitCoilName",
    "receive_coil":                "ReceiveCoilName",
    "sar":                         "SAR",
    "db_dt":                       "dBdt",

    # DTI / Diffusion
    "diffusion_directionality":    "DiffusionDirectionality",
    "b_value":                     "DiffusionBValue",
    "diffusion_gradient_orientation": "DiffusionGradientOrientation",
    "anisotropy_type":             "AnisotropyType",

    # CT-specific
    "kvp":                         "KVP",
    "tube_current_mA":             "XRayTubeCurrent",
    "exposure_time_ms":            "ExposureTime",
    "exposure_mAs":                "Exposure",
    "ctdi_vol":                    "CTDIvol",
    "convolution_kernel":          "ConvolutionKernel",
    "data_collection_diameter_mm": "DataCollectionDiameter",

    # Series file information (always populated)
    "series_dir":                  None,
    "series_files":                None,
}


def _safe_get(ds: pydicom.Dataset, keyword: str) -> Any:
    """Return the *value* for *keyword* from *ds*, or ``None`` if absent/error."""
    try:
        elem = ds[keyword]
        val = elem.value
        if isinstance(val, pydicom.multival.MultiValue):
            return list(val)
        if hasattr(val, "__class__") and val.__class__.__name__ == "PersonName":
            return str(val)
        return val
    except (KeyError, AttributeError):
        return None


def _extract_file_row(ds: pydicom.Dataset, file_path: Path) -> dict[str, Any]:
    """Extract all configured tags from a single DICOM dataset.

    Returns a dict keyed by the column names in :data:`DICOM_TAGS` plus an
    internal ``_file_path`` entry used for series grouping.  The derived
    columns (``number_of_slices``, ``series_dir``, ``series_files``) are
    left as ``None`` here and filled during series aggregation.
    """
    row: dict[str, Any] = {}

    for col, keyword in DICOM_TAGS.items():
        if keyword is None:
            row[col] = None  # derived – filled during aggregation
        else:
            row[col] = _safe_get(ds, keyword)

    # Derived: split PixelSpacing
    pixel_spacing = row.get("pixel_spacing")
    if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2:
        try:
            row["pixel_spacing_row_mm"] = float(pixel_spacing[0])
            row["pixel_spacing_col_mm"] = float(pixel_spacing[1])
        except (ValueError, TypeError):
            pass
    elif isinstance(pixel_spacing, str):
        parts = pixel_spacing.split("\\")
        if len(parts) >= 2:
            try:
                row["pixel_spacing_row_mm"] = float(parts[0])
                row["pixel_spacing_col_mm"] = float(parts[1])
            except (ValueError, TypeError):
                pass

    # Derived: field of view
    try:
        rows_val = int(row["rows"]) if row["rows"] is not None else None
        ps_row = row["pixel_spacing_row_mm"]
        if rows_val is not None and ps_row is not None:
            row["field_of_view_mm"] = rows_val * float(ps_row)
    except (TypeError, ValueError):
        pass

    # Store pixel_spacing as a plain string for DataFrame compatibility
    if isinstance(pixel_spacing, list):
        row["pixel_spacing"] = "\\".join(str(v) for v in pixel_spacing)

    # Internal field used only for grouping – not exposed in output
    row["_file_path"] = str(file_path)

    return row


def _series_key(row: dict[str, Any]) -> str:
    """Return a grouping key for *row*.

    Prefers ``SeriesInstanceUID`` for correctness; falls back to the parent
    directory of the file so that old DICOM files without a UID are still
    grouped sensibly.
    """
    uid = row.get("series_instance_uid")
    if uid:
        return str(uid)
    return str(Path(row["_file_path"]).parent)


def _is_null(val: Any) -> bool:
    """Return True if *val* should be treated as absent/null."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


def _aggregate_series(file_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a list of per-file metadata dicts into one series-level row.

    For each metadata column the first non-null value found across all files
    in the series is used.  File paths are collected into a sorted list
    (``series_files``) and the common directory is stored in ``series_dir``.
    """
    result: dict[str, Any] = {}

    metadata_cols = [
        col for col in DICOM_TAGS
        if col not in ("number_of_slices", "series_dir", "series_files")
    ]

    for col in metadata_cols:
        result[col] = None
        for row in file_rows:
            val = row.get(col)
            if not _is_null(val):
                result[col] = val
                break

    # Derived: file list, directory, number of slices
    all_paths = sorted(row["_file_path"] for row in file_rows)
    result["series_files"] = all_paths
    result["series_dir"] = str(Path(all_paths[0]).parent) if all_paths else None
    result["number_of_slices"] = len(all_paths)

    return result


def load_dicom_metadata(
    root: str | os.PathLike,
    *,
    glob_pattern: str = "**/*",
    force: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Recursively scan *root* for DICOM files and return a per-series DataFrame.

    DICOM files are discovered recursively under *root*, grouped by
    ``SeriesInstanceUID`` (or by folder when the UID is absent), and
    aggregated so that **each row represents exactly one scan / series**.
    Slice-specific attributes (instance number, position, etc.) are not
    included; instead the full list of file paths for each series is stored
    in the ``series_files`` column and the count in ``number_of_slices``.

    Parameters
    ----------
    root:
        Path to the top-level folder that contains DICOM files (possibly spread
        across many sub-folders).
    glob_pattern:
        Glob pattern used to enumerate candidate files under *root*.
        Defaults to ``"**/*"`` which matches every file recursively.
    force:
        Pass ``force=True`` to ``pydicom.dcmread`` so that files without the
        standard DICOM preamble are also attempted.  Useful for older scanners.
    show_progress:
        If ``True`` and the optional *tqdm* package is installed, a progress
        bar is displayed while scanning.

    Returns
    -------
    pandas.DataFrame
        One row per DICOM series (scan).  Columns are defined by
        :data:`DICOM_TAGS`.  All columns are nullable; absent tags produce
        ``None`` / ``NaN``.  The ``series_files`` column contains a Python
        list of absolute file paths that belong to that series.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    NotADirectoryError
        If *root* exists but is not a directory.

    Examples
    --------
    >>> from brainmaze_imaging.dicom import load_dicom_metadata
    >>> df = load_dicom_metadata("/data/dicoms")
    >>> print(df[["subject_id", "series_description", "number_of_slices"]])
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"DICOM root directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root_path}")

    candidate_files = [
        p for p in root_path.glob(glob_pattern) if p.is_file()
    ]

    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            candidate_files = tqdm(candidate_files, desc="Reading DICOM files", unit="file")
        except ImportError:
            logger.warning("tqdm not installed; progress bar unavailable.")

    file_rows: list[dict[str, Any]] = []
    skipped = 0

    for file_path in candidate_files:
        try:
            ds = pydicom.dcmread(
                str(file_path),
                stop_before_pixels=True,
                force=force,
            )
            file_rows.append(_extract_file_row(ds, file_path))
        except InvalidDicomError:
            skipped += 1
            logger.debug("Skipping non-DICOM file: %s", file_path)
        except (OSError, PermissionError, ValueError) as exc:
            skipped += 1
            logger.warning("Error reading %s: %s", file_path, exc)

    if skipped:
        logger.info("Skipped %d non-DICOM / unreadable files.", skipped)

    if not file_rows:
        logger.warning("No DICOM files found under %s", root_path)
        return pd.DataFrame(columns=list(DICOM_TAGS.keys()))

    # Group by series and aggregate to one row per series
    series_groups: dict[str, list[dict[str, Any]]] = {}
    for row in file_rows:
        key = _series_key(row)
        series_groups.setdefault(key, []).append(row)

    series_rows = [_aggregate_series(group) for group in series_groups.values()]

    df = pd.DataFrame(series_rows)

    # Ensure all expected columns are present (even if all NaN)
    for col in DICOM_TAGS:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns to match DICOM_TAGS definition order
    ordered_cols = [c for c in DICOM_TAGS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in DICOM_TAGS]
    df = df[ordered_cols + extra_cols]

    return df.reset_index(drop=True)
