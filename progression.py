"""
KLERT Tumor Progression Analysis & DICOM Visualizer Module
Reads DICOM series from Brain-Tumor-Progression dataset (PGBM-XXX patients),
computes 3D tumor volumes from MaskTumor series, renders peak-tumor cross-section
MRI slice images with colored mask contours, and calculates longitudinal change analysis.
"""

import base64
import glob
import io
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Dataset root ─────────────────────────────────────────────────────────────
_DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", "./data/Brain-Tumor-Progression")
).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_mask_volume(series_dir: Path) -> tuple[np.ndarray, float, float, float]:
    """Load all DICOM slices from a MaskTumor series directory.
    Returns (3D numpy array, pixel_spacing_row, pixel_spacing_col, slice_thickness).
    """
    files = sorted(glob.glob(str(series_dir / "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")

    slices = []
    ps_row, ps_col, thickness = 1.0, 1.0, 1.0

    for fpath in files:
        ds = pydicom.dcmread(fpath, force=True)
        slices.append(ds.pixel_array.astype(np.uint8))
        if hasattr(ds, "PixelSpacing"):
            ps_row = float(ds.PixelSpacing[0])
            ps_col = float(ds.PixelSpacing[1])
        if hasattr(ds, "SliceThickness"):
            thickness = float(ds.SliceThickness)

    volume = np.stack(slices, axis=0)  # shape: (slices, rows, cols)
    return volume, ps_row, ps_col, thickness


def _compute_volume_cm3(mask_vol: np.ndarray, ps_row: float, ps_col: float, thickness: float) -> float:
    """Compute tumor volume in cm³ from binary mask."""
    voxel_mm3 = ps_row * ps_col * thickness
    tumor_voxels = int(np.sum(mask_vol > 0))
    return (tumor_voxels * voxel_mm3) / 1000.0


def _find_series(session_dir: Path, kind: str) -> Optional[Path]:
    """Find a series subfolder matching a keyword (e.g. 'MaskTumor', 'T1post')."""
    for sub in sorted(session_dir.iterdir()):
        if sub.is_dir() and kind.lower() in sub.name.lower():
            return sub
    return None


def _parse_date_from_dirname(dirname: str) -> Optional[str]:
    """Extract date string like 11-19-1991 from session folder names."""
    match = re.match(r"(\d{2}-\d{2}-\d{4})", dirname)
    return match.group(1) if match else None


def _render_dicom_slice_overlay(t1_file: str, mask_file: str) -> str:
    """
    Reads DICOM MRI slice + Tumor Mask slice, normalizes intensity,
    blends red translucent tumor highlight and neon contour line,
    and returns a base64 PNG data URL string.
    """
    try:
        t1_ds = pydicom.dcmread(t1_file, force=True)
        m_ds = pydicom.dcmread(mask_file, force=True)

        img_arr = t1_ds.pixel_array.astype(float)
        # Min-max normalization for DICOM intensity
        min_v, max_v = img_arr.min(), img_arr.max()
        if max_v > min_v:
            img_norm = ((img_arr - min_v) / (max_v - min_v) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img_arr, dtype=np.uint8)

        rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        mask_arr = (m_ds.pixel_array > 0).astype(np.uint8)

        # Translucent red overlay on tumor region
        overlay = rgb.copy()
        overlay[mask_arr > 0] = [245, 55, 95]  # Neon crimson red

        blended = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

        # Draw contour line around tumor boundary
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 60, 120), 2)

        pil_img = Image.fromarray(blended)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as e:
        print(f"[!] Error rendering slice overlay: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API used by server.py
# ─────────────────────────────────────────────────────────────────────────────

def list_patients() -> list[str]:
    """List all PGBM patient IDs available in the dataset."""
    if not _DATASET_ROOT.exists():
        return []
    return sorted(
        [d.name for d in _DATASET_ROOT.iterdir() if d.is_dir() and d.name.startswith("PGBM")]
    )


def get_patient_sessions(patient_id: str) -> list[dict]:
    """Return list of session dicts for a patient."""
    patient_dir = _DATASET_ROOT / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient {patient_id} not found in dataset")

    sessions = []
    for session_dir in sorted(patient_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        date_str = _parse_date_from_dirname(session_dir.name)
        series_list = [s.name for s in sorted(session_dir.iterdir()) if s.is_dir()]

        has_mask = any("masktumor" in s.lower() for s in series_list)
        has_t1 = any("t1post" in s.lower() for s in series_list)

        sessions.append({
            "session_folder": session_dir.name,
            "date": date_str,
            "has_mask": has_mask,
            "has_t1post": has_t1,
            "series": series_list,
        })

    return sessions


def compute_progression(patient_id: str) -> dict:
    """
    Compute longitudinal tumor volume progression for a patient.
    Finds all sessions with MaskTumor series, computes 3D volumes, renders
    the peak-tumor DICOM cross-section MRI image, and returns delta/trajectory data.
    """
    patient_dir = _DATASET_ROOT / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient {patient_id} not found")

    timepoints = []

    for session_dir in sorted(patient_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        mask_series = _find_series(session_dir, "MaskTumor")
        if mask_series is None:
            continue

        date_str = _parse_date_from_dirname(session_dir.name) or session_dir.name

        try:
            mask_vol, ps_row, ps_col, thickness = _load_mask_volume(mask_series)
            vol_cm3 = _compute_volume_cm3(mask_vol, ps_row, ps_col, thickness)

            # Find slice with peak tumor area
            slice_areas = [int(np.sum(mask_vol[i] > 0)) for i in range(mask_vol.shape[0])]
            peak_slice_idx = int(np.argmax(slice_areas)) if max(slice_areas) > 0 else mask_vol.shape[0] // 2
            slices_with_tumor = int(np.sum(np.array(slice_areas) > 0))

            t1_series = _find_series(session_dir, "T1post")

            # Render peak slice overlay if T1 series exists
            slice_image_b64 = ""
            if t1_series:
                t1_files = sorted(glob.glob(str(t1_series / "*.dcm")))
                mask_files = sorted(glob.glob(str(mask_series / "*.dcm")))
                if t1_files and mask_files:
                    idx = min(peak_slice_idx, len(t1_files) - 1, len(mask_files) - 1)
                    slice_image_b64 = _render_dicom_slice_overlay(t1_files[idx], mask_files[idx])

            timepoints.append({
                "date": date_str,
                "session": session_dir.name,
                "volume_cm3": round(vol_cm3, 3),
                "tumor_voxels": int(np.sum(mask_vol > 0)),
                "slices_with_tumor": slices_with_tumor,
                "total_slices": mask_vol.shape[0],
                "peak_slice_index": peak_slice_idx + 1,
                "pixel_spacing_mm": [round(ps_row, 4), round(ps_col, 4)],
                "slice_thickness_mm": round(thickness, 4),
                "mask_series": mask_series.name,
                "t1_series": t1_series.name if t1_series else None,
                "slice_image_b64": slice_image_b64,
            })
        except Exception as e:
            timepoints.append({
                "date": date_str,
                "session": session_dir.name,
                "error": str(e),
                "volume_cm3": None,
                "slice_image_b64": "",
            })

    if not timepoints:
        return {
            "patient_id": patient_id,
            "timepoints": [],
            "summary": {"error": "No MaskTumor series found for this patient."},
        }

    # ── Sort timepoints chronologically ──
    # Re-order by date if parsed
    timepoints = sorted(
        timepoints,
        key=lambda x: (
            int(x["date"].split("-")[2]) if x["date"] and len(x["date"].split("-")) == 3 else 0,
            int(x["date"].split("-")[0]) if x["date"] and len(x["date"].split("-")) == 3 else 0,
            int(x["date"].split("-")[1]) if x["date"] and len(x["date"].split("-")) == 3 else 0,
        )
    )

    # ── Compute longitudinal delta ──
    valid = [t for t in timepoints if t.get("volume_cm3") is not None]

    summary = {}
    if len(valid) >= 2:
        baseline_vol = valid[0]["volume_cm3"]
        latest_vol = valid[-1]["volume_cm3"]
        delta_cm3 = latest_vol - baseline_vol
        pct_change = (delta_cm3 / baseline_vol * 100) if baseline_vol > 0 else 0

        if pct_change > 20:
            trajectory = "TUMOR PROGRESSION (GROWTH)"
            trajectory_color = "danger"
        elif pct_change < -20:
            trajectory = "TUMOR REGRESSION (SHRINKAGE)"
            trajectory_color = "success"
        else:
            trajectory = "STABLE DISEASE"
            trajectory_color = "warning"

        intervals = []
        for i in range(1, len(valid)):
            prev = valid[i - 1]
            curr = valid[i]
            d = curr["volume_cm3"] - prev["volume_cm3"]
            pct = (d / prev["volume_cm3"] * 100) if prev["volume_cm3"] > 0 else 0
            intervals.append({
                "from": prev["date"],
                "to": curr["date"],
                "delta_cm3": round(d, 3),
                "pct_change": round(pct, 1),
            })

        summary = {
            "patient_id": patient_id,
            "n_timepoints": len(valid),
            "baseline_date": valid[0]["date"],
            "latest_date": valid[-1]["date"],
            "baseline_volume_cm3": baseline_vol,
            "latest_volume_cm3": latest_vol,
            "total_delta_cm3": round(delta_cm3, 3),
            "total_pct_change": round(pct_change, 1),
            "trajectory": trajectory,
            "trajectory_color": trajectory_color,
            "intervals": intervals,
        }
    elif len(valid) == 1:
        summary = {
            "patient_id": patient_id,
            "n_timepoints": 1,
            "baseline_date": valid[0]["date"],
            "baseline_volume_cm3": valid[0]["volume_cm3"],
            "trajectory": "SINGLE TIMEPOINT",
            "trajectory_color": "secondary",
        }

    return {
        "patient_id": patient_id,
        "timepoints": timepoints,
        "summary": summary,
    }
