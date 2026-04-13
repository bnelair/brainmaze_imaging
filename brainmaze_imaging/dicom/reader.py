"""
DICOM metadata reader.

Recursively walks a directory tree and returns a :class:`pandas.DataFrame`
where **each row represents one scan / series** (one directory of DICOM files).

For efficiency the reader processes only **one representative DICOM file per
series directory** to extract metadata.  The total number of files in that
directory is reported as ``number_of_slices`` and a fast size estimate
(``series_size_bytes_estimate``) is computed as
``number_of_slices × representative_file_size``.  This makes the function
practical even for datasets with hundreds of scans and thousands of slices.

The tag selection follows the same conventions used by **dcm2niix** so that the
resulting DataFrame is immediately useful for quality-control workflows and for
driving DICOM-to-NIfTI conversions.

Typical usage
-------------
>>> from brainmaze_imaging.dicom import load_dicom_metadata
>>> df = load_dicom_metadata("/path/to/dicom/root")
>>> print(df[["subject_id", "series_description", "number_of_slices", "series_size_bytes_estimate"]])
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
    "number_of_slices":            None,
    "series_size_bytes_estimate":  None,
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


def _extract_series_row(ds: pydicom.Dataset) -> dict[str, Any]:
    """Extract all configured tags from a representative DICOM dataset.

    The derived columns (``number_of_slices``, ``series_dir``,
    ``series_size_bytes_estimate``) are left as ``None`` here and filled
    by the caller after counting files and measuring disk usage.
    """
    row: dict[str, Any] = {}

    for col, keyword in DICOM_TAGS.items():
        if keyword is None:
            row[col] = None  # derived – filled by caller
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

    return row


def _read_representative(
    candidates: list[Path],
    force: bool = False,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Try files in sorted order until a valid DICOM is found.

    Returns ``(row_dict, representative_path)`` on success, or
    ``(None, None)`` if no valid DICOM exists among *candidates*.
    Only the headers are read (``stop_before_pixels=True``), keeping
    I/O minimal.
    """
    for path in sorted(candidates):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=force)
            return _extract_series_row(ds), path
        except (InvalidDicomError, OSError, PermissionError, ValueError):
            continue
    return None, None


def load_dicom_metadata(
    root: str | os.PathLike,
    *,
    glob_pattern: str = "**/*",
    force: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Recursively scan *root* for DICOM files and return a per-series DataFrame.

    Files are grouped by their **parent directory** – one row is produced per
    directory that contains at least one valid DICOM file.  Only a **single
    representative file** (the first valid DICOM in sorted filename order) is
    parsed for metadata, making the function fast even for large datasets.

    ``number_of_slices`` is the total count of regular files in that directory
    (a fast OS-level count, no file parsing).  ``series_size_bytes_estimate``
    is ``number_of_slices × representative_file_size``.

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
        bar is displayed while scanning directories.

    Returns
    -------
    pandas.DataFrame
        One row per DICOM series directory.  Columns are defined by
        :data:`DICOM_TAGS`.  All columns are nullable; absent tags produce
        ``None`` / ``NaN``.

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
    >>> print(df[["subject_id", "series_description", "number_of_slices", "series_size_bytes_estimate"]])
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"DICOM root directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root_path}")

    # ── Group candidate files by parent directory (no file reads yet) ────────
    dir_files: dict[Path, list[Path]] = {}
    for p in root_path.glob(glob_pattern):
        if p.is_file():
            dir_files.setdefault(p.parent, []).append(p)

    if not dir_files:
        logger.warning("No files found under %s", root_path)
        return pd.DataFrame(columns=list(DICOM_TAGS.keys()))

    directories = list(dir_files.keys())
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            directories = tqdm(directories, desc="Reading series directories", unit="dir")
        except ImportError:
            logger.warning("tqdm not installed; progress bar unavailable.")

    series_rows: list[dict[str, Any]] = []
    skipped_dirs = 0

    for dir_path in directories:
        candidates = dir_files[dir_path]
        row, rep_path = _read_representative(candidates, force=force)

        if row is None:
            skipped_dirs += 1
            logger.debug("No valid DICOM found in %s", dir_path)
            continue

        n_files = len(candidates)
        rep_size = rep_path.stat().st_size  # type: ignore[union-attr]

        row["series_dir"] = str(dir_path)
        row["number_of_slices"] = n_files
        row["series_size_bytes_estimate"] = n_files * rep_size

        series_rows.append(row)

    if skipped_dirs:
        logger.info("Skipped %d directories with no valid DICOM files.", skipped_dirs)

    if not series_rows:
        logger.warning("No DICOM series found under %s", root_path)
        return pd.DataFrame(columns=list(DICOM_TAGS.keys()))

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
