"""Unit tests for brainmaze_imaging.dicom.reader.

Synthetic in-memory DICOM datasets are created with pydicom so that no real
DICOM files are required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from brainmaze_imaging.dicom.reader import DICOM_TAGS, load_dicom_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mr_dataset(
    patient_id: str = "SUB001",
    patient_name: str = "Doe^John",
    series_uid: str | None = None,
    instance_number: int = 1,
    series_description: str = "MPRAGE",
    **extra_tags,
) -> FileDataset:
    """Return a minimal synthetic MR DICOM dataset."""
    if series_uid is None:
        series_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"
    ds.PatientAge = "044Y"

    # Study / Series
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = generate_uid()
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number

    # Dates
    ds.AcquisitionDate = "20240101"
    ds.AcquisitionTime = "120000.000"
    ds.SeriesDate = "20240101"
    ds.StudyDate = "20240101"
    ds.SeriesTime = "120000.000"

    # Equipment
    ds.Manufacturer = "Siemens"
    ds.ManufacturerModelName = "Prisma"
    ds.MagneticFieldStrength = 3.0

    # Modality / Sequence
    ds.Modality = "MR"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.SeriesDescription = series_description
    ds.ProtocolName = series_description
    ds.SequenceName = "*tfl3d1"
    ds.ScanningSequence = "GR"
    ds.SequenceVariant = "SP"
    ds.ScanOptions = "FS"

    # Geometry
    ds.Rows = 256
    ds.Columns = 256
    ds.SliceThickness = 1.0
    ds.SpacingBetweenSlices = 1.0
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(instance_number)]
    ds.SliceLocation = float(instance_number)

    # MRI timing
    ds.RepetitionTime = 2000.0
    ds.EchoTime = 2.93
    ds.InversionTime = 900.0
    ds.FlipAngle = 9.0
    ds.EchoTrainLength = 1

    # MRI acquisition parameters
    ds.NumberOfAverages = 1.0
    ds.PercentSampling = 100.0
    ds.PercentPhaseFieldOfView = 100.0
    ds.PixelBandwidth = 240.0
    ds.InPlanePhaseEncodingDirection = "ROW"
    ds.NumberOfPhaseEncodingSteps = 256

    # Parallel imaging
    ds.ParallelReductionFactorInPlane = 2.0

    # Coils
    ds.TransmitCoilName = "Body"
    ds.ReceiveCoilName = "Head_64"

    # Apply any extra keyword arguments
    for kw, val in extra_tags.items():
        setattr(ds, kw, val)

    return ds


def _make_ct_dataset(
    patient_id: str = "SUB002",
    series_uid: str | None = None,
    instance_number: int = 1,
) -> FileDataset:
    """Return a minimal synthetic CT DICOM dataset."""
    if series_uid is None:
        series_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.PatientID = patient_id
    ds.PatientName = "Smith^Jane"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.AcquisitionDate = "20240202"
    ds.Modality = "CT"
    ds.SeriesDescription = "HEAD ROUTINE"
    ds.Rows = 512
    ds.Columns = 512
    ds.SliceThickness = 2.5
    ds.PixelSpacing = [0.488, 0.488]
    ds.KVP = 120.0
    ds.XRayTubeCurrent = 300
    ds.ConvolutionKernel = "H30s"

    return ds


def _write_dicom(ds: FileDataset, path: Path) -> None:
    """Save *ds* to *path* using pydicom's save_as."""
    ds.save_as(str(path), enforce_file_format=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDicomMetadata:
    """Tests for load_dicom_metadata()."""

    def test_empty_directory_returns_empty_dataframe(self, tmp_path):
        df = load_dicom_metadata(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # All expected columns must be present
        for col in DICOM_TAGS:
            assert col in df.columns, f"Missing column: {col}"

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dicom_metadata(tmp_path / "does_not_exist")

    def test_file_path_raises_not_a_directory(self, tmp_path):
        f = tmp_path / "somefile.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            load_dicom_metadata(f)

    def test_single_mr_file(self, tmp_path):
        ds = _make_mr_dataset(patient_id="P001", series_description="MPRAGE")
        _write_dicom(ds, tmp_path / "slice1.dcm")

        df = load_dicom_metadata(tmp_path)

        assert len(df) == 1
        assert df["subject_id"].iloc[0] == "P001"
        assert df["modality"].iloc[0] == "MR"
        assert df["series_description"].iloc[0] == "MPRAGE"

    def test_mr_file_mri_parameters_present(self, tmp_path):
        ds = _make_mr_dataset()
        _write_dicom(ds, tmp_path / "slice1.dcm")

        df = load_dicom_metadata(tmp_path)

        assert float(df["repetition_time_ms"].iloc[0]) == pytest.approx(2000.0)
        assert float(df["echo_time_ms"].iloc[0]) == pytest.approx(2.93)
        assert float(df["inversion_time_ms"].iloc[0]) == pytest.approx(900.0)
        assert float(df["flip_angle_deg"].iloc[0]) == pytest.approx(9.0)
        assert float(df["nex"].iloc[0]) == pytest.approx(1.0)
        assert df["phase_encoding_direction"].iloc[0] == "ROW"
        assert float(df["parallel_reduction_factor"].iloc[0]) == pytest.approx(2.0)

    def test_pixel_spacing_derived_columns(self, tmp_path):
        ds = _make_mr_dataset()
        _write_dicom(ds, tmp_path / "s.dcm")

        df = load_dicom_metadata(tmp_path)

        assert float(df["pixel_spacing_row_mm"].iloc[0]) == pytest.approx(1.0)
        assert float(df["pixel_spacing_col_mm"].iloc[0]) == pytest.approx(1.0)

    def test_field_of_view_derived(self, tmp_path):
        ds = _make_mr_dataset()
        _write_dicom(ds, tmp_path / "s.dcm")

        df = load_dicom_metadata(tmp_path)

        # FOV = rows * pixel_spacing_row = 256 * 1.0 = 256 mm
        assert float(df["field_of_view_mm"].iloc[0]) == pytest.approx(256.0)

    def test_number_of_slices_per_series(self, tmp_path):
        series_uid = generate_uid()
        for i in range(5):
            ds = _make_mr_dataset(series_uid=series_uid, instance_number=i + 1)
            _write_dicom(ds, tmp_path / f"slice{i}.dcm")

        df = load_dicom_metadata(tmp_path)

        assert len(df) == 5
        assert (df["number_of_slices"] == 5).all()

    def test_multiple_series_slice_count(self, tmp_path):
        uid_a = generate_uid()
        uid_b = generate_uid()

        for i in range(3):
            ds = _make_mr_dataset(series_uid=uid_a, instance_number=i + 1)
            _write_dicom(ds, tmp_path / f"seriesA_{i}.dcm")

        for i in range(7):
            ds = _make_mr_dataset(series_uid=uid_b, instance_number=i + 1,
                                   series_description="FGATIR")
            _write_dicom(ds, tmp_path / f"seriesB_{i}.dcm")

        df = load_dicom_metadata(tmp_path)

        assert len(df) == 10
        counts = (
            df.groupby("series_instance_uid")["number_of_slices"]
            .first()
            .sort_values()
            .tolist()
        )
        assert counts == [3, 7]

    def test_ct_file(self, tmp_path):
        ds = _make_ct_dataset(patient_id="P999")
        _write_dicom(ds, tmp_path / "ct.dcm")

        df = load_dicom_metadata(tmp_path)

        assert df["modality"].iloc[0] == "CT"
        assert df["series_description"].iloc[0] == "HEAD ROUTINE"
        assert float(df["kvp"].iloc[0]) == pytest.approx(120.0)
        assert df["convolution_kernel"].iloc[0] == "H30s"

    def test_non_dicom_files_are_skipped(self, tmp_path):
        (tmp_path / "not_a_dicom.txt").write_text("hello")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        ds = _make_mr_dataset(patient_id="P001")
        _write_dicom(ds, tmp_path / "real.dcm")

        df = load_dicom_metadata(tmp_path)
        assert len(df) == 1

    def test_nested_subdirectories(self, tmp_path):
        sub = tmp_path / "subject01" / "session01" / "T1"
        sub.mkdir(parents=True)

        ds = _make_mr_dataset(patient_id="NESTED01")
        _write_dicom(ds, sub / "slice1.dcm")

        df = load_dicom_metadata(tmp_path)

        assert len(df) == 1
        assert df["subject_id"].iloc[0] == "NESTED01"
        assert "subject01" in df["file_path"].iloc[0]

    def test_all_expected_columns_present(self, tmp_path):
        ds = _make_mr_dataset()
        _write_dicom(ds, tmp_path / "s.dcm")

        df = load_dicom_metadata(tmp_path)

        for col in DICOM_TAGS:
            assert col in df.columns, f"Missing column: {col}"

    def test_missing_tags_are_none_or_nan(self, tmp_path):
        """Tags not present in a file should not raise – they become NaN/None."""
        ds = _make_mr_dataset()
        # Remove a tag that is normally present
        if hasattr(ds, "InversionTime"):
            del ds.InversionTime

        _write_dicom(ds, tmp_path / "s.dcm")

        df = load_dicom_metadata(tmp_path)

        val = df["inversion_time_ms"].iloc[0]
        assert val is None or (isinstance(val, float) and np.isnan(val))

    def test_mixed_modalities(self, tmp_path):
        mr_dir = tmp_path / "MR"
        ct_dir = tmp_path / "CT"
        mr_dir.mkdir()
        ct_dir.mkdir()

        _write_dicom(_make_mr_dataset(patient_id="MR01"), mr_dir / "mr.dcm")
        _write_dicom(_make_ct_dataset(patient_id="CT01"), ct_dir / "ct.dcm")

        df = load_dicom_metadata(tmp_path)

        assert len(df) == 2
        assert set(df["modality"].tolist()) == {"MR", "CT"}

    def test_returns_dataframe_type(self, tmp_path):
        ds = _make_mr_dataset()
        _write_dicom(ds, tmp_path / "s.dcm")
        df = load_dicom_metadata(tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_file_path_column_is_absolute(self, tmp_path):
        _write_dicom(_make_mr_dataset(), tmp_path / "s.dcm")
        df = load_dicom_metadata(tmp_path)
        assert os.path.isabs(df["file_path"].iloc[0])

    def test_dti_b_value(self, tmp_path):
        ds = _make_mr_dataset(series_description="DTI_64DIR")
        ds.DiffusionBValue = 1000
        ds.DiffusionDirectionality = "BMATRIX"
        _write_dicom(ds, tmp_path / "dti.dcm")

        df = load_dicom_metadata(tmp_path)

        assert int(df["b_value"].iloc[0]) == 1000
        assert df["diffusion_directionality"].iloc[0] == "BMATRIX"

    def test_accepts_string_and_pathlib_root(self, tmp_path):
        _write_dicom(_make_mr_dataset(), tmp_path / "s.dcm")

        df_str = load_dicom_metadata(str(tmp_path))
        df_path = load_dicom_metadata(tmp_path)

        assert len(df_str) == len(df_path) == 1
