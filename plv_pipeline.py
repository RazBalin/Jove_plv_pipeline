#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plv_pipeline.py
===============

Phase-Locking Value (PLV) analysis pipeline for the murine Default Mode Network
(DMN) electrophysiology dataset accompanying

    Raz et al. "Electrophysiological characterisation of the murine Default Mode
    Network across healthy, depression-model, and LSD-plasticity states".
    Journal of Visualized Experiments (JoVE), in revision.

This script is the *exact* program used to produce the PLV connectivity matrices
reported in the paper and supplementary figures. It is distributed as a
stand-alone pipeline on purpose: reviewers asked for a single reproducible
artefact, and keeping the published code minimal reduces the attack surface for
post-publication critique.

The pipeline computes two classes of output from complex Morlet-wavelet
coefficient files (`*_raw-tfr-complex.h5`):

    (A) Individual-subject PLV matrices
        -- one matrix per (subject × probe × frequency band), saved as both
           .npz and .png plus a tidy CSV of upper-triangle edge values.

    (B) Group-level PLV matrices + between-group comparison
        -- group-averaged PLV matrix per (group × probe × band), subject
           variance, N; plus, when two groups are specified, a between-group
           independent-samples t-statistic matrix with Benjamini--Hochberg
           (BH, 1995) FDR correction on the off-diagonal edge family.

No other analyses are implemented here. For wPLI, PAC, FOOOF, Granger, NBS,
and graph-theoretic analyses please refer to the full Pipelines4/ repository
cited in the paper.

Scientific conventions
----------------------
*   PLV estimator (Lachaux et al., 1999):

        PLV_{ij} = | (1/T) * sum_{t=1..T}  exp( i * ( phi_i(t) - phi_j(t) ) ) |

    where phi is the instantaneous phase extracted from the complex Morlet
    coefficient z(t) = A(t) exp(i phi(t)) as angle(z). Implemented natively
    with NumPy; no crosspy/MNE PLV dependency.

*   Unbiased estimator -- Pairwise Phase Consistency (PPC; Vinck et al., 2010,
    NeuroImage 51:112-122):

        PPC_{ij} = (T * PLV_{ij}^2 - 1) / (T - 1)

    PLV is biased upward ~ 1/sqrt(T); PPC removes the leading-order sample-size
    bias. Both metrics are saved. Figures in the paper use PLV to match the
    historical notation, but PPC is the recommended metric for quantitative
    comparison across recordings of different length.

*   Band integration. For each band we identify the subset of canonical
    wavelet frequencies in-band, compute one PLV matrix per frequency, and
    average the matrices (equivalent to averaging the per-frequency squared
    coherence estimates under the assumption of independent-in-frequency
    phase-locking, which is the standard practice in the literature).

*   Edge buffer. We discard the first and last `edge_buffer_s` seconds of each
    epoch to mitigate the wavelet cone-of-influence artefact. Default is 2 s,
    consistent with the wavelet decomposition settings in the upstream
    preprocessing (Bhatt et al., 2025, bioRxiv 10.1101/2025.11.13.688186v2).

*   Region aggregation. Multi-channel probes (Neuropixels ProbeA / ProbeB and
    the 4-node eCoG grid) are collapsed to anatomical regions by averaging
    complex coefficients across all channels assigned to the same region.
    Averaging is performed on the *complex* coefficients before phase
    extraction, which is equivalent to a phase-weighted mean and preserves
    the well-defined phase of the regional LFP.

*   Group statistics. Between-group edge-wise Welch's t-tests on the
    upper-triangle of each PLV matrix, BH-FDR corrected at q=0.05 across the
    C(n_regions, 2) edge family. Cohen's d (pooled SD) is reported for
    effect-size interpretation.

File-name conventions (matching the JoVE dataset)
-------------------------------------------------
    {AnimalID}_{Probe}_{Group}_raw-tfr-complex.h5
      where  AnimalID  = rb1 ... rb99            (lowercase in filename)
             Probe     = ProbeA | ProbeB | FPGA  (case-insensitive substring)
             Group     = BL | MDD | LSD | BL-ctrl | MDD-ctrl | LSD-ctrl

Usage
-----
    python plv_pipeline.py                                  # interactive menu
    python plv_pipeline.py --mode individual                # all subjects, all bands
    python plv_pipeline.py --mode group --groups BL MDD     # 2-group comparison
    python plv_pipeline.py --mode group --groups BL         # single-group mean

Outputs are written to a ``Results/`` sub-folder next to this script. The
pipeline is self-contained: configuration lives in the ``Config`` dataclass at
the top of this file; no command-line flags are *required* (defaults are the
ones used for the paper).

Dependencies
------------
    numpy, scipy, h5py, matplotlib, statsmodels (>=0.13 for multipletests)

Python 3.9+ tested on Ubuntu 22.04 and Windows 11.

Author
------
    Raz Bhatt, 2026 (PhD candidate, Castrén lab, University of Helsinki)
    For questions: raz.bhatt@helsinki.fi (or open an issue on the repository).

Licence
-------
    MIT (see LICENSE).
"""

from __future__ import annotations

# ---- Standard library ------------------------------------------------------
import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# ---- Scientific stack ------------------------------------------------------
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Module-level logger. We configure it in main() so that imports are quiet.
# ---------------------------------------------------------------------------
log = logging.getLogger("plv_pipeline")


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
# A single ``@dataclass`` collects every tuneable parameter. All pipeline
# functions read their defaults from a ``Config`` instance that is passed
# explicitly -- no magic globals, no hidden state. Any of these fields can be
# overridden from a custom driver script without touching this file.
# ===========================================================================

@dataclass
class Config:
    """Immutable-by-convention configuration for the PLV pipeline."""

    # --- Data acquisition ---------------------------------------------------
    sample_rate_hz: float = 250.0
    """Downsampled eCoG / Neuropixels sample rate after preprocessing."""

    epoch_start_s: float = 0.0
    """Start time (in seconds) of the analysis window inside each recording."""

    epoch_end_s: float = 120.0
    """End time (in seconds); default is the full 2-minute artefact-free epoch."""

    edge_buffer_s: float = 2.0
    """Seconds discarded at each end of the epoch to suppress wavelet COI."""

    # --- Wavelet frequency grid --------------------------------------------
    # The 35-frequency vector below is the *exact* log-spaced grid used by the
    # upstream wavelet decomposition that produces ``*_raw-tfr-complex.h5``.
    # If you change the upstream grid you must change this vector in lock-step,
    # otherwise the band-to-index mapping will be silently wrong.
    default_freqs_hz: np.ndarray = field(default_factory=lambda: np.array([
        3,      3.68,   4.0227, 4.4419, 5.0536, 5.8133, 6.5622, 7.402,
        8.0455, 8.8659, 10.086, 11.552, 13.061, 14.887, 16,     17.633,
        20.095, 23.1,   25.952, 30,     35.429, 40,     45,     51.429,
        60,     72,     80,     90,     102.86, 120,    144,    160,
        180,    204,    240,
    ], dtype=np.float64))

    # --- Canonical frequency bands -----------------------------------------
    # Half-open convention: a frequency f belongs to band (lo, hi) if
    #     lo <= f <= hi.
    # These match the bands in Pipelines4/ and the JoVE manuscript.
    frequency_bands_hz: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Delta":     (1.0,   4.0),
        "Theta":     (4.0,   8.0),
        "Alpha":     (8.0,  12.0),
        "Beta":     (12.0,  30.0),
        "LowGamma": (30.0,  60.0),
        "HighGamma":(60.0, 100.0),
    })

    # --- Statistics --------------------------------------------------------
    fdr_alpha: float = 0.05
    """BH-FDR q-value threshold for edge-wise significance."""

    min_subjects_per_group: int = 2
    """Hard floor below which group statistics are skipped (not reported)."""

    use_ppc: bool = True
    """If True, additionally save the bias-corrected PPC metric (Vinck 2010)."""

    # --- File I/O ----------------------------------------------------------
    data_dir: Optional[str] = None
    """Directory containing ``*_raw-tfr-complex.h5`` files. If None, defaults
    to the directory this script lives in (see ``resolve_paths``)."""

    output_dir: Optional[str] = None
    """Directory where Results/ will be written. Defaults to the script dir."""

    figure_dpi: int = 300
    figure_format: str = "png"   # also supports "svg" and "pdf"

    random_seed: int = 42
    """Seed for any stochastic component (only used by permutation tests)."""


# ===========================================================================
# 2. FILENAME PARSING
# ===========================================================================
# The dataset uses a simple filename convention. We centralise the regex logic
# here so that a future refactor to pathlib-based metadata is a one-line change.
# ===========================================================================

# Mapping order matters: longer patterns must be checked BEFORE shorter
# substrings, otherwise "bl-ctrl" would be mis-classified as "bl".
_GROUP_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("bl-ctrl",  "BL-ctrl"),
    ("blctrl",   "BL-ctrl"),
    ("mdd-ctrl", "MDD-ctrl"),
    ("mddctrl",  "MDD-ctrl"),
    ("lsd-ctrl", "LSD-ctrl"),
    ("lsdctrl",  "LSD-ctrl"),
    ("bl",       "BL"),
    ("mdd",      "MDD"),
    ("lsd",      "LSD"),
)


def identify_group(filename: str) -> str:
    """Return the canonical group label from a filename.

    The filename is first normalised (lowercased; underscores / punctuation
    collapsed to single spaces) so that cosmetic differences in the recording
    software's naming do not change the outcome. Returns ``"Unknown"`` if no
    pattern matches.
    """
    base = os.path.splitext(os.path.basename(filename))[0].lower().strip()
    base = re.sub(r"[\W_]+", " ", base)       # "rb10_ProbeA_BL-ctrl" -> "rb10 probea bl ctrl"
    tokens = " " + base + " "                 # add sentinels for cleaner matching
    # First pass: long forms that include "ctrl"
    for needle, label in _GROUP_PATTERNS[:6]:
        if f" {needle} " in tokens or needle.replace("-", " ") in tokens:
            return label
    # Second pass: short forms
    for needle, label in _GROUP_PATTERNS[6:]:
        if f" {needle} " in tokens:
            return label
    return "Unknown"


def identify_probe(filename: str) -> str:
    """Return 'ProbeA', 'ProbeB', or 'FPGA' (eCoG) from a filename."""
    fname_lower = os.path.basename(filename).lower()
    if "probea" in fname_lower:
        return "ProbeA"
    if "probeb" in fname_lower:
        return "ProbeB"
    # Default to eCoG (FPGA). Files without an explicit probe tag in the
    # upstream naming convention are FPGA / eCoG recordings.
    return "FPGA"


def identify_animal(filename: str) -> str:
    """Return the animal ID ('rbNN') from a filename, or 'unknown'."""
    m = re.search(r"(rb[0-9]+)", os.path.basename(filename).lower())
    return m.group(1) if m else "unknown"


# ===========================================================================
# 3. ELECTRODE -> REGION MAPPINGS
# ===========================================================================
# These mappings assign each raw channel number to its anatomical region based
# on the probe geometry and the histology-verified surface/column layout.
# They are the *exact* mappings used in Pipelines4/ and the JoVE manuscript.
# Each channel not found in a map is labelled "Unknown" and will be dropped
# from downstream PLV computation (so the matrices always use anatomical
# region labels, never raw channel numbers).
# ===========================================================================

def _build_probe_b_mapping() -> Dict[str, str]:
    """Neuropixels ProbeB (mPFC column, 0..383)."""
    m: Dict[str, str] = {}
    # Dorsal-most cortex
    m.update({f"Ch_{i}": "MOs1"     for i in range(355, 384)})
    m.update({f"Ch_{i}": "MOs2/3"   for i in range(300, 354)})
    m.update({f"Ch_{i}": "MOs5"     for i in range(270, 299)})
    m.update({f"Ch_{i}": "MOs6a"    for i in range(240, 269)})
    m.update({f"Ch_{i}": "ACA6a"    for i in range(220, 239)})
    m.update({f"Ch_{i}": "PL6a"     for i in range(200, 219)})
    m.update({f"Ch_{i}": "PL5"      for i in range(180, 199)})
    m.update({f"Ch_{i}": "ILA5"     for i in range(160, 179)})
    # ILA2/3 is non-contiguous in the column
    m.update({f"Ch_{i}": "ILA2/3"   for i in list(range(140, 159)) + list(range(100, 119))})
    m.update({f"Ch_{i}": "ILA1"     for i in range(120, 139)})
    m.update({f"Ch_{i}": "ORBm2/3"  for i in list(range(90, 99)) + list(range(50, 79))})
    m.update({f"Ch_{i}": "ORBm5"    for i in range(80, 89)})
    m.update({f"Ch_{i}": "ORBvl2/3" for i in range(30, 49)})
    m.update({f"Ch_{i}": "OLF"      for i in range(0, 29)})
    return m


def _build_probe_a_mapping() -> Dict[str, str]:
    """Neuropixels ProbeA (hippocampal-thalamic-visual column, 0..383)."""
    m: Dict[str, str] = {}
    m.update({f"Ch_{i}": "VISam1"   for i in range(365, 384)})
    m.update({f"Ch_{i}": "VISam2/3" for i in range(346, 364)})
    m.update({f"Ch_{i}": "VISam4"   for i in range(336, 345)})
    m.update({f"Ch_{i}": "VISam5"   for i in range(326, 335)})
    m.update({f"Ch_{i}": "VISam6a"  for i in range(307, 325)})
    m.update({f"Ch_{i}": "APN6b"    for i in range(298, 306)})
    m.update({f"Ch_{i}": "CCS"      for i in range(259, 278)})
    m.update({f"Ch_{i}": "CA1"      for i in range(230, 258)})
    m.update({f"Ch_{i}": "DG-mo"    for i in range(211, 229)})
    m.update({f"Ch_{i}": "DG-sg"    for i in range(202, 210)})
    m.update({f"Ch_{i}": "DGcr-po"  for i in range(192, 201)})
    m.update({f"Ch_{i}": "DG-mo-Rt" for i in range(173, 181)})
    m.update({f"Ch_{i}": "TH-LP"    for i in range(125, 172)})
    m.update({f"Ch_{i}": "TH-PO"    for i in range(58, 124)})
    m.update({f"Ch_{i}": "TH-VPM"   for i in range(0, 57)})
    return m


def _build_ecog_mapping() -> Dict[str, str]:
    """FPGA / eCoG 4-node DMN surface grid."""
    m: Dict[str, str] = {}
    # Right anterior cingulate area
    for i in [3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 37, 38, 39, 40, 41, 42,
              54, 55, 56, 57, 58, 59, 71, 72, 73, 74, 75, 76,
              88, 89, 90, 91, 92, 93, 105, 106, 107, 108, 109, 110,
              122, 123, 124, 125, 126, 127,
              133, 134, 135, 136, 137, 138, 139,
              156, 157, 158, 159, 160, 161,
              173, 174, 175, 176, 177, 178]:
        m[f"Ch_{i}"] = "R-ACA"
    # Left anterior cingulate area
    for i in [8, 9, 10, 11, 12, 13, 26, 27, 28, 29, 30, 31, 43, 44, 45, 46,
              60, 61, 62, 63, 64, 65, 77, 78, 79, 80, 81,
              94, 95, 96, 97, 98, 99, 111, 112, 113, 114, 115,
              128, 129, 130, 131, 132,
              145, 146, 147, 148, 162, 163, 164, 165, 166, 167,
              179, 180, 181, 182]:
        m[f"Ch_{i}"] = "L-ACA"
    # Right retrosplenial
    for i in [256, 257, 258, 259, 260, 261, 262, 263, 264,
              273, 274, 275, 276, 277, 278, 279, 280,
              289, 290, 291, 292, 293, 294, 295, 296, 297,
              305, 306, 307, 308, 309, 310, 311, 312,
              321, 322, 323, 324, 325, 326, 327, 328, 329,
              337, 338, 339, 340, 341, 342, 343, 344,
              353, 354, 355, 356, 357, 358, 359, 360, 361,
              369, 370, 371, 372, 373, 374, 375, 376]:
        m[f"Ch_{i}"] = "R-RSP"
    # Left retrosplenial
    for i in [265, 266, 267, 268, 269, 270, 271, 272,
              281, 282, 283, 284, 285, 286, 287, 288,
              298, 299, 300, 301, 302, 303, 304,
              313, 314, 315, 316, 317, 318, 319, 320,
              330, 331, 332, 333, 334, 335, 336,
              345, 346, 347, 348, 349, 350, 351, 352,
              362, 363, 364, 365, 366, 367, 368,
              377, 378, 379, 380, 381, 382, 383, 384,
              394, 395, 396, 397, 398, 399, 400,
              409, 410, 411, 412, 413, 414, 415, 416,
              426, 427, 428, 429, 430, 431, 432,
              441, 442, 443, 444, 445, 446, 447, 448,
              458, 459, 460, 461, 462, 463, 464]:
        m[f"Ch_{i}"] = "L-RSP"
    return m


PROBE_B_MAPPING = _build_probe_b_mapping()
PROBE_A_MAPPING = _build_probe_a_mapping()
ECOG_MAPPING    = _build_ecog_mapping()


def probe_mapping_for(probe_type: str) -> Dict[str, str]:
    if probe_type == "ProbeA":
        return PROBE_A_MAPPING
    if probe_type == "ProbeB":
        return PROBE_B_MAPPING
    return ECOG_MAPPING


# ===========================================================================
# 4. DATA LOADING
# ===========================================================================

def load_complex_tfr(filepath: str, cfg: Config) -> Optional[Dict[str, np.ndarray]]:
    """Load a ``*_raw-tfr-complex.h5`` file and collapse channels to regions.

    The HDF5 file must contain a dataset ``wavelet_complex`` of shape
    ``(n_channels, n_frequencies, n_timepoints)`` and dtype ``complex128``
    (or any compatible complex dtype). The canonical preprocessing pipeline
    in Pipelines4/ always produces this layout.

    Returns a dict ``{region: 2-D complex array [n_freq, n_time], ...,
                      "freqs": 1-D array [n_freq]}``,
    or ``None`` if the file could not be read.
    """
    try:
        with h5py.File(filepath, "r") as h5f:
            if "wavelet_complex" not in h5f:
                log.error("Dataset 'wavelet_complex' not found in %s", filepath)
                return None
            wavelet_data = h5f["wavelet_complex"][()]  # force read
    except Exception as exc:
        log.error("Could not read %s: %s", filepath, exc)
        return None

    # Expected shape: (n_ch, n_freq, n_time). We do *not* assume the upstream
    # wavelet grid length equals cfg.default_freqs_hz -- just truncate if the
    # file contains fewer frequencies than the default grid.
    if wavelet_data.ndim != 3:
        log.error("Unexpected array shape %s in %s (expected 3-D)",
                  wavelet_data.shape, filepath)
        return None
    n_ch, n_freq, n_time = wavelet_data.shape
    if n_freq > cfg.default_freqs_hz.shape[0]:
        log.warning("File %s has %d wavelet frequencies; pipeline default is %d."
                    " Extra frequencies will be silently dropped.",
                    filepath, n_freq, cfg.default_freqs_hz.shape[0])
        wavelet_data = wavelet_data[:, :cfg.default_freqs_hz.shape[0], :]
        n_freq = wavelet_data.shape[1]

    probe_type = identify_probe(filepath)
    region_map = probe_mapping_for(probe_type)

    # Collapse channels to regions by complex mean. Averaging BEFORE phase
    # extraction is the phase-preserving aggregation used throughout the
    # project -- this matches the behaviour of Pipelines4/ load_complex_tfr.
    region_buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    for ch_idx in range(n_ch):
        region = region_map.get(f"Ch_{ch_idx}", "Unknown")
        region_buckets[region].append(wavelet_data[ch_idx])

    region_dict: Dict[str, np.ndarray] = {}
    for region, arr_list in region_buckets.items():
        if region == "Unknown":
            # Drop unmapped channels; they are not part of any DMN node.
            continue
        region_dict[region] = np.mean(np.stack(arr_list, axis=0), axis=0)

    region_dict["freqs"] = cfg.default_freqs_hz[:n_freq].copy()
    return region_dict


# ===========================================================================
# 5. PLV CORE COMPUTATION
# ===========================================================================

def _slice_complex_signal(region_data_2d: np.ndarray,
                          freq_idx: int,
                          cfg: Config) -> np.ndarray:
    """Extract the 1-D complex time-series for one region at one frequency,
    applying the epoch window and the wavelet-COI edge buffer.

    The edge buffer drops ``edge_buffer_s`` seconds from both ends of the
    requested epoch. If the buffered window would be empty or negative, we
    fall back to the full window and log a warning.
    """
    fs = cfg.sample_rate_hz
    start_samp = int(round((cfg.epoch_start_s + cfg.edge_buffer_s) * fs))
    end_samp   = int(round((cfg.epoch_end_s   - cfg.edge_buffer_s) * fs))
    if end_samp <= start_samp:
        log.warning("Edge buffer (%.2fs) too large for epoch [%.1f, %.1f]s; "
                    "using the full window with no buffer.",
                    cfg.edge_buffer_s, cfg.epoch_start_s, cfg.epoch_end_s)
        start_samp = int(round(cfg.epoch_start_s * fs))
        end_samp   = int(round(cfg.epoch_end_s   * fs))

    n_freq_available = region_data_2d.shape[0]
    if freq_idx >= n_freq_available:
        # Silent clamp: the upstream wavelet grid is sometimes shorter than
        # the default -- we clip to the last valid frequency rather than
        # raising, since the downstream code averages over band indices.
        freq_idx = n_freq_available - 1

    sig = region_data_2d[freq_idx, start_samp:end_samp]
    # Truncate to same length across regions (the slice is deterministic, so
    # this is just defensive programming against upstream shape drift).
    return np.ascontiguousarray(sig)


def compute_plv_matrix(region_dict: Dict[str, np.ndarray],
                       freq_idx: int,
                       cfg: Config) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute the single-frequency PLV matrix for one subject/recording.

    Parameters
    ----------
    region_dict
        Output of ``load_complex_tfr``; must contain complex 2-D arrays under
        anatomical-region keys plus the 1-D ``freqs`` entry.
    freq_idx
        Index into the wavelet frequency grid.
    cfg
        Pipeline configuration (sample rate, epoch, edge buffer, PPC flag).

    Returns
    -------
    region_list
        Sorted list of region labels (alphabetical, "freqs" excluded).
    plv_matrix
        Symmetric ``n_regions x n_regions`` float32 array in [0, 1].
    ppc_matrix
        Symmetric ``n_regions x n_regions`` float32 PPC matrix in
        [-1/(T-1), 1]. Values below zero indicate below-chance phase
        consistency and are returned as-is (NOT clipped), so that the
        unbiased estimator remains interpretable.

    Notes
    -----
    Implementation uses the unit-norm-phase-sum identity:

        PLV_{ij} = | mean_t( u_i(t) * conj(u_j(t)) ) |
                 = | <u_i, u_j> / T |

    where u_k(t) = exp(i*angle(z_k(t))) is the unit-modulus phase vector.
    This avoids a nested Python loop over timepoints and runs in O(T) per
    edge with pure NumPy BLAS.
    """
    # 1. Sort regions so that matrix ordering is deterministic and
    #    reproducible across runs / subjects.
    region_list = sorted([r for r in region_dict.keys() if r != "freqs"])
    n_r = len(region_list)
    if n_r < 2:
        log.warning("Fewer than 2 regions in this recording (%d); "
                    "returning an empty matrix.", n_r)
        return region_list, np.zeros((n_r, n_r), dtype=np.float32), \
               np.zeros((n_r, n_r), dtype=np.float32)

    # 2. Build the unit-phase matrix U of shape (n_r, T).
    phase_vecs: List[np.ndarray] = []
    for region in region_list:
        sig = _slice_complex_signal(region_dict[region], freq_idx, cfg)
        amp = np.abs(sig)
        # Guard against the (essentially impossible) zero-amplitude sample.
        amp = np.where(amp > 0, amp, 1.0)
        phase_vecs.append(sig / amp)
    U = np.stack(phase_vecs, axis=0)                 # shape (n_r, T)
    T = U.shape[1]
    if T < 4:
        log.warning("Too few time samples (%d) after edge-buffering; "
                    "PLV is undefined.", T)
        return region_list, np.zeros((n_r, n_r), dtype=np.float32), \
               np.zeros((n_r, n_r), dtype=np.float32)

    # 3. Vectorised PLV: the (i,j) entry is |<U_i, conj(U_j)>| / T.
    #    U @ U.conj().T yields exactly this inner-product matrix.
    inner = (U @ U.conj().T) / T                     # complex (n_r, n_r)
    plv_mat = np.abs(inner).astype(np.float32)
    # Numerical housekeeping: the diagonal is analytically 1, force it.
    np.fill_diagonal(plv_mat, 1.0)

    # 4. Pairwise Phase Consistency (bias-corrected; Vinck et al. 2010):
    #        PPC = (T * PLV^2 - 1) / (T - 1)
    ppc_mat = ((T * plv_mat.astype(np.float64) ** 2 - 1.0)
               / float(T - 1)).astype(np.float32)
    np.fill_diagonal(ppc_mat, 1.0)

    return region_list, plv_mat, ppc_mat


def compute_plv_band_matrix(region_dict: Dict[str, np.ndarray],
                            freq_indices: Sequence[int],
                            cfg: Config) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """PLV matrix averaged across a set of wavelet-grid frequencies.

    A separate PLV matrix is computed at each frequency index in
    ``freq_indices`` and the resulting matrices are averaged element-wise.
    This is the standard "band-integrated PLV" operation used in the paper.

    If no valid index is found (e.g. the caller asks for a band that does
    not intersect the available wavelet frequencies), we fall back to the
    last available frequency and emit a warning -- never raise -- so that
    bulk batch runs do not abort mid-pipeline.
    """
    # Determine the largest valid index for this file.
    max_freq_idx = region_dict["freqs"].shape[0] if "freqs" in region_dict else None
    if max_freq_idx is None:
        # Fall back to a region's first dimension length.
        some_region = next((r for r in region_dict.keys() if r != "freqs"), None)
        max_freq_idx = region_dict[some_region].shape[0] if some_region else 1

    valid = [int(i) for i in freq_indices if 0 <= int(i) < max_freq_idx]
    if not valid:
        log.warning("No valid frequency indices for this band; "
                    "using last available index %d.", max_freq_idx - 1)
        valid = [max_freq_idx - 1]

    plv_list: List[np.ndarray] = []
    ppc_list: List[np.ndarray] = []
    region_list: List[str] = []
    for idx in valid:
        region_list, plv_mat, ppc_mat = compute_plv_matrix(region_dict, idx, cfg)
        plv_list.append(plv_mat)
        ppc_list.append(ppc_mat)

    avg_plv = np.mean(np.stack(plv_list, axis=0), axis=0)
    avg_ppc = np.mean(np.stack(ppc_list, axis=0), axis=0)
    return region_list, avg_plv, avg_ppc


def freq_indices_for_band(band: Tuple[float, float], cfg: Config) -> np.ndarray:
    """Return the wavelet-grid indices inside the closed band interval."""
    lo, hi = band
    return np.where((cfg.default_freqs_hz >= lo) & (cfg.default_freqs_hz <= hi))[0]


# ===========================================================================
# 6. MATRIX ALIGNMENT ACROSS SUBJECTS
# ===========================================================================
# Different recording sessions may drop individual channels (bad electrodes),
# so the region list per subject can vary. We align all matrices to the union
# of regions; missing cells become NaN and are excluded from means / stats
# via NaN-aware functions.
# ===========================================================================

def align_connectivity_matrix(region_list: Sequence[str],
                              matrix: np.ndarray,
                              common_regions: Sequence[str]) -> np.ndarray:
    """Re-index a connectivity matrix onto the ``common_regions`` ordering."""
    idx_map = {r: i for i, r in enumerate(region_list)}
    n = len(common_regions)
    aligned = np.full((n, n), np.nan, dtype=np.float32)
    for i, r1 in enumerate(common_regions):
        if r1 not in idx_map:
            continue
        for j, r2 in enumerate(common_regions):
            if r2 not in idx_map:
                continue
            aligned[i, j] = matrix[idx_map[r1], idx_map[r2]]
    return aligned


def aggregate_group(subject_matrices: List[np.ndarray],
                    subject_region_lists: List[List[str]]
                    ) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
    """Group-level mean matrix with per-edge variance across subjects."""
    if not subject_matrices:
        return [], np.zeros((0, 0)), np.zeros((0, 0)), 0
    common = sorted(set().union(*subject_region_lists))
    aligned = [align_connectivity_matrix(r, m, common)
               for r, m in zip(subject_region_lists, subject_matrices)]
    stack = np.stack(aligned, axis=0)                  # (N, R, R)
    mean_mat = np.nanmean(stack, axis=0)
    var_mat  = np.nanvar(stack, axis=0, ddof=1)
    return common, mean_mat, var_mat, len(subject_matrices)


# ===========================================================================
# 7. GROUP COMPARISON STATISTICS
# ===========================================================================

def edgewise_group_comparison(group_a_mats: List[np.ndarray],
                              group_b_mats: List[np.ndarray],
                              common_regions: List[str],
                              cfg: Config) -> Dict[str, np.ndarray]:
    """Edge-wise independent-samples Welch t-test with BH-FDR correction.

    The test family is the upper triangle (k=1) of the n_regions x n_regions
    matrix. Diagonal entries are by construction identically 1 and are not
    tested. Missing cells (NaN) are dropped from the t-test via ``nan_policy=
    'omit'`` and then from the FDR family altogether.

    Returns
    -------
    dict with keys:
        t_matrix           (n x n)   -- signed t-values
        p_matrix           (n x n)   -- raw p-values (NaN on diagonal / missing)
        p_fdr_matrix       (n x n)   -- BH-FDR q-values
        reject_matrix      (n x n)   -- bool, q < alpha
        cohens_d_matrix    (n x n)   -- pooled-SD effect size
    """
    n = len(common_regions)
    a = np.stack(group_a_mats, axis=0)
    b = np.stack(group_b_mats, axis=0)

    # Element-wise Welch t-test with nan_policy='omit'.
    t_stat, p_raw = stats.ttest_ind(a, b, axis=0,
                                    equal_var=False, nan_policy="omit")
    # Cast to dense numpy (ttest_ind with nan_policy returns a masked array).
    t_stat = np.asarray(t_stat, dtype=np.float64)
    p_raw  = np.asarray(p_raw,  dtype=np.float64)
    # The masked entries become the sentinel `1.0 - 0.0j`-style values; we
    # replace those with NaN explicitly for clarity.
    if hasattr(t_stat, "mask"):
        t_stat = np.where(t_stat.mask, np.nan, t_stat.data)
    if hasattr(p_raw, "mask"):
        p_raw  = np.where(p_raw.mask,  np.nan, p_raw.data)

    # Cohen's d with pooled SD (Welch variant is uncommon; we report the
    # classical pooled-SD d, which is interpretable under the reviewer-
    # expected convention).
    mean_a = np.nanmean(a, axis=0); mean_b = np.nanmean(b, axis=0)
    var_a  = np.nanvar(a, axis=0, ddof=1)
    var_b  = np.nanvar(b, axis=0, ddof=1)
    n_a = np.sum(~np.isnan(a), axis=0)
    n_b = np.sum(~np.isnan(b), axis=0)
    denom = np.where((n_a + n_b - 2) > 0, n_a + n_b - 2, 1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / denom
    pooled_sd  = np.sqrt(np.maximum(pooled_var, 1e-12))
    cohens_d   = (mean_a - mean_b) / pooled_sd

    # BH-FDR over the upper-triangle edge family. We deliberately exclude
    # the diagonal (which is analytically 1 in PLV) and any NaN entries.
    triu = np.triu_indices(n, k=1)
    p_vec = p_raw[triu]
    valid = ~np.isnan(p_vec)
    p_fdr = np.full(p_vec.shape, np.nan, dtype=np.float64)
    reject = np.zeros(p_vec.shape, dtype=bool)
    if valid.any():
        rej, p_corr, _, _ = multipletests(p_vec[valid],
                                          alpha=cfg.fdr_alpha,
                                          method="fdr_bh")
        p_fdr[valid]  = p_corr
        reject[valid] = rej

    p_fdr_mat  = np.full((n, n), np.nan, dtype=np.float64)
    reject_mat = np.zeros((n, n), dtype=bool)
    p_fdr_mat[triu]  = p_fdr
    reject_mat[triu] = reject
    # Mirror to lower triangle so downstream users get a symmetric matrix.
    p_fdr_mat  = np.where(np.isnan(p_fdr_mat),  p_fdr_mat.T,  p_fdr_mat)
    reject_mat = reject_mat | reject_mat.T

    return {
        "t_matrix":        t_stat,
        "p_matrix":        p_raw,
        "p_fdr_matrix":    p_fdr_mat,
        "reject_matrix":   reject_mat,
        "cohens_d_matrix": cohens_d,
    }


# ===========================================================================
# 8. PLOTTING
# ===========================================================================

def plot_connectivity_matrix(matrix: np.ndarray,
                             region_list: Sequence[str],
                             title: str,
                             savepath: str,
                             vmin: float = 0.0,
                             vmax: float = 1.0,
                             cmap: str = "viridis",
                             significance_mask: Optional[np.ndarray] = None,
                             cbar_label: str = "PLV",
                             dpi: int = 300) -> None:
    """Publication-quality heatmap of a connectivity matrix.

    Parameters
    ----------
    significance_mask
        Optional boolean mask of the same shape as ``matrix``; True cells are
        marked with a white asterisk ("*"). Use for FDR-surviving edges.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(range(len(region_list)))
    ax.set_yticks(range(len(region_list)))
    ax.set_xticklabels(region_list, rotation=90, fontsize=8)
    ax.set_yticklabels(region_list, fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    if significance_mask is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if significance_mask[i, j] and i != j:
                    ax.text(j, i, "*", ha="center", va="center",
                            color="white", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 9. I/O HELPERS
# ===========================================================================

def sanitise_filename(s: str) -> str:
    """Strip characters that are invalid in file names on Windows/Unix."""
    s = re.sub(r"[<>:\"/\\|?*\n\r]", "", s)
    s = s.replace(" ", "_")
    return s


def save_matrix_artefacts(matrix: np.ndarray,
                          region_list: Sequence[str],
                          stem: str,
                          output_dir: str,
                          matrix_kind: str = "PLV") -> None:
    """Persist a connectivity matrix as (.npz + .csv).

    The .npz payload contains ``matrix`` and ``regions`` so that downstream
    scripts can round-trip without re-parsing the CSV. The CSV contains the
    upper-triangle edge list (region_a, region_b, value), the tidy format
    preferred for ingestion into R/Pandas.
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{stem}.npz")
    csv_path = os.path.join(output_dir, f"{stem}.csv")

    np.savez_compressed(npz_path, matrix=matrix, regions=np.array(region_list, dtype=object))
    n = len(region_list)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["region_a", "region_b", matrix_kind])
        for i in range(n):
            for j in range(i + 1, n):
                writer.writerow([region_list[i], region_list[j], f"{matrix[i, j]:.6f}"])


# ===========================================================================
# 10. HIGH-LEVEL PIPELINE ENTRY POINTS
# ===========================================================================

def resolve_paths(cfg: Config) -> Tuple[str, str]:
    """Return (data_dir, output_dir), defaulting to the script directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = cfg.data_dir   or script_dir
    output_dir = os.path.join(cfg.output_dir or script_dir, "Results")
    os.makedirs(output_dir, exist_ok=True)
    return data_dir, output_dir


def discover_complex_tfr_files(data_dir: str) -> List[str]:
    """Non-recursive glob for ``*_raw-tfr-complex.h5`` files in ``data_dir``."""
    pat = os.path.join(data_dir, "*_raw-tfr-complex.h5")
    files = sorted(glob.glob(pat))
    if not files:
        # Try the case-insensitive fallback for Windows-origin file names.
        pat2 = os.path.join(data_dir, "*raw-tfr-complex*")
        files = sorted(f for f in glob.glob(pat2)
                       if f.lower().endswith(".h5"))
    return files


# ---- (A) Individual-subject PLV -------------------------------------------

def run_individual_pipeline(cfg: Config,
                            file_list: Optional[List[str]] = None) -> None:
    """Compute and save PLV matrices for every (file × band) in ``file_list``.

    If ``file_list`` is None, all ``*_raw-tfr-complex.h5`` files in the data
    directory are processed. One .npz + .csv + .png is saved per (subject,
    band) combination under Results/individual/.
    """
    data_dir, out_dir = resolve_paths(cfg)
    out_dir = os.path.join(out_dir, "individual")
    os.makedirs(out_dir, exist_ok=True)

    files = file_list if file_list is not None else discover_complex_tfr_files(data_dir)
    if not files:
        log.error("No *_raw-tfr-complex.h5 files found in %s", data_dir)
        return

    log.info("Individual PLV analysis: %d file(s), %d band(s).",
             len(files), len(cfg.frequency_bands_hz))

    # Top-level summary CSV with one row per (subject, band, edge).
    summary_path = os.path.join(out_dir, "individual_plv_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["animal_id", "group", "probe", "band",
                         "region_a", "region_b", "plv", "ppc"])

        for filepath in files:
            fname = os.path.basename(filepath)
            animal = identify_animal(fname)
            group  = identify_group(fname)
            probe  = identify_probe(fname)
            log.info("  [%s | %s | %s] %s", animal, group, probe, fname)

            region_dict = load_complex_tfr(filepath, cfg)
            if region_dict is None:
                log.warning("  Skipped: could not load.")
                continue

            for band_name, band_range in cfg.frequency_bands_hz.items():
                freq_idx = freq_indices_for_band(band_range, cfg)
                if freq_idx.size == 0:
                    log.warning("  No wavelet frequencies inside %s band %s.",
                                band_name, band_range)
                    continue

                region_list, plv_mat, ppc_mat = compute_plv_band_matrix(
                    region_dict, freq_idx, cfg)
                if not region_list:
                    log.warning("  No regions -- skipping %s.", band_name)
                    continue

                stem = sanitise_filename(
                    f"individual_{animal}_{probe}_{group}_{band_name}_PLV")
                save_matrix_artefacts(plv_mat, region_list, stem, out_dir, "PLV")
                if cfg.use_ppc:
                    save_matrix_artefacts(
                        ppc_mat, region_list,
                        stem.replace("_PLV", "_PPC"), out_dir, "PPC")

                title = (f"{animal} [{probe}] - {group} - {band_name} band\n"
                         f"Phase-Locking Value ({band_range[0]:g}-{band_range[1]:g} Hz)")
                plot_connectivity_matrix(
                    plv_mat, region_list, title,
                    savepath=os.path.join(
                        out_dir, f"{stem}.{cfg.figure_format}"),
                    vmin=0.0, vmax=1.0, cmap="viridis",
                    cbar_label="PLV",
                    dpi=cfg.figure_dpi)

                for i, r1 in enumerate(region_list):
                    for j, r2 in enumerate(region_list):
                        if j <= i:
                            continue
                        writer.writerow([animal, group, probe, band_name,
                                         r1, r2,
                                         f"{plv_mat[i, j]:.6f}",
                                         f"{ppc_mat[i, j]:.6f}"])
    log.info("Individual PLV analysis complete. Summary: %s", summary_path)


# ---- (B) Group-level PLV ---------------------------------------------------

def run_group_pipeline(cfg: Config,
                       group_labels: Sequence[str],
                       file_list: Optional[List[str]] = None) -> None:
    """Group-level PLV mean matrices and (optionally) between-group comparison.

    If one group is passed, produce the mean matrix only (per probe × band).
    If two groups are passed, additionally produce the edge-wise Welch t-test
    matrix with BH-FDR.
    """
    if not (1 <= len(group_labels) <= 2):
        raise ValueError("group_labels must have length 1 or 2; got "
                         f"{len(group_labels)}: {group_labels}")

    data_dir, out_dir = resolve_paths(cfg)
    out_dir = os.path.join(out_dir, "group")
    os.makedirs(out_dir, exist_ok=True)

    files = file_list if file_list is not None else discover_complex_tfr_files(data_dir)
    if not files:
        log.error("No *_raw-tfr-complex.h5 files found in %s", data_dir)
        return

    # Bucket files: group_label -> probe -> list of (filepath, animal_id).
    buckets: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list))
    for fp in files:
        g = identify_group(fp)
        p = identify_probe(fp)
        a = identify_animal(fp)
        if g in group_labels:
            buckets[g][p].append((fp, a))

    # Pre-compute per-subject band matrices only once, cached in a nested
    # dict: group -> probe -> band -> list of (region_list, plv, ppc, animal)
    cache: Dict[str, Dict[str, Dict[str, List[Tuple[List[str], np.ndarray, np.ndarray, str]]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for group in group_labels:
        for probe, flist in buckets[group].items():
            for fp, animal in flist:
                region_dict = load_complex_tfr(fp, cfg)
                if region_dict is None:
                    continue
                for band_name, band_range in cfg.frequency_bands_hz.items():
                    idx = freq_indices_for_band(band_range, cfg)
                    if idx.size == 0:
                        continue
                    r_list, plv, ppc = compute_plv_band_matrix(
                        region_dict, idx, cfg)
                    if r_list:
                        cache[group][probe][band_name].append((r_list, plv, ppc, animal))

    # Walk the cache and produce figures/CSVs.
    all_probes = set()
    for g in group_labels:
        all_probes.update(cache[g].keys())

    comparison_summary_rows: List[Dict] = []

    for probe in sorted(all_probes):
        for band_name in cfg.frequency_bands_hz.keys():
            group_means: Dict[str, Tuple[List[str], np.ndarray, np.ndarray, int]] = {}
            group_subject_mats: Dict[str, List[np.ndarray]] = {}
            group_subject_regions: Dict[str, List[List[str]]] = {}

            for group in group_labels:
                entries = cache[group][probe][band_name]
                if not entries:
                    log.info("[%s | %s | %s] no subjects; skipping.",
                             group, probe, band_name)
                    continue
                region_lists = [e[0] for e in entries]
                plv_mats     = [e[1] for e in entries]
                common, mean_mat, var_mat, n_subj = aggregate_group(
                    plv_mats, region_lists)
                group_means[group] = (common, mean_mat, var_mat, n_subj)
                group_subject_mats[group] = plv_mats
                group_subject_regions[group] = region_lists

                # Save the group-mean matrix figure and CSV.
                stem = sanitise_filename(
                    f"group_{group}_{probe}_{band_name}_meanPLV_n{n_subj}")
                save_matrix_artefacts(mean_mat, common, stem, out_dir, "PLV")
                title = (f"Group {group} [{probe}] - {band_name} band\n"
                         f"Mean PLV (n = {n_subj} subjects)")
                plot_connectivity_matrix(
                    mean_mat, common, title,
                    savepath=os.path.join(out_dir, f"{stem}.{cfg.figure_format}"),
                    vmin=0.0, vmax=1.0, cmap="viridis",
                    cbar_label="PLV", dpi=cfg.figure_dpi)

            # If exactly two groups and both present at this (probe, band),
            # run the between-group comparison.
            if len(group_labels) == 2 \
                    and all(g in group_means for g in group_labels):
                g_a, g_b = group_labels
                # Align both groups to the same union of regions.
                union = sorted(set(group_means[g_a][0])
                               .union(group_means[g_b][0]))
                a_mats = [align_connectivity_matrix(r, m, union)
                          for r, m in zip(group_subject_regions[g_a],
                                          group_subject_mats[g_a])]
                b_mats = [align_connectivity_matrix(r, m, union)
                          for r, m in zip(group_subject_regions[g_b],
                                          group_subject_mats[g_b])]

                if (len(a_mats) < cfg.min_subjects_per_group
                        or len(b_mats) < cfg.min_subjects_per_group):
                    log.info("  [%s vs %s | %s | %s] too few subjects "
                             "(n_a=%d, n_b=%d); skipping comparison.",
                             g_a, g_b, probe, band_name,
                             len(a_mats), len(b_mats))
                    continue

                stats_res = edgewise_group_comparison(
                    a_mats, b_mats, union, cfg)

                stem = sanitise_filename(
                    f"group_{g_a}_vs_{g_b}_{probe}_{band_name}_tstat")
                save_matrix_artefacts(stats_res["t_matrix"], union,
                                      stem, out_dir, "t")
                save_matrix_artefacts(stats_res["cohens_d_matrix"], union,
                                      stem.replace("_tstat", "_cohens_d"),
                                      out_dir, "d")
                save_matrix_artefacts(stats_res["p_fdr_matrix"], union,
                                      stem.replace("_tstat", "_pfdr"),
                                      out_dir, "p_fdr")

                abs_t = np.nanmax(np.abs(stats_res["t_matrix"]))
                vmax  = abs_t if np.isfinite(abs_t) and abs_t > 0 else 1.0
                title = (f"{g_a} vs {g_b} [{probe}] - {band_name} band\n"
                         f"PLV t-statistic (n_a={len(a_mats)}, "
                         f"n_b={len(b_mats)}; * = q_FDR<{cfg.fdr_alpha:g})")
                plot_connectivity_matrix(
                    stats_res["t_matrix"], union, title,
                    savepath=os.path.join(out_dir, f"{stem}.{cfg.figure_format}"),
                    vmin=-vmax, vmax=vmax, cmap="coolwarm",
                    cbar_label="t (Welch)",
                    significance_mask=stats_res["reject_matrix"],
                    dpi=cfg.figure_dpi)

                # Collect a row per surviving edge for the summary CSV.
                n = len(union)
                for i in range(n):
                    for j in range(i + 1, n):
                        if np.isnan(stats_res["p_fdr_matrix"][i, j]):
                            continue
                        comparison_summary_rows.append({
                            "group_a": g_a, "group_b": g_b,
                            "probe":  probe, "band":    band_name,
                            "region_a": union[i], "region_b": union[j],
                            "mean_a": float(np.nanmean(
                                [m[i, j] for m in a_mats])),
                            "mean_b": float(np.nanmean(
                                [m[i, j] for m in b_mats])),
                            "t": float(stats_res["t_matrix"][i, j]),
                            "p_raw": float(stats_res["p_matrix"][i, j]),
                            "p_fdr": float(stats_res["p_fdr_matrix"][i, j]),
                            "cohens_d": float(stats_res["cohens_d_matrix"][i, j]),
                            "reject_fdr": bool(stats_res["reject_matrix"][i, j]),
                            "n_a": len(a_mats), "n_b": len(b_mats),
                        })

    # Write per-edge comparison summary (only when two groups were requested).
    if len(group_labels) == 2 and comparison_summary_rows:
        summary_path = os.path.join(out_dir, "group_comparison_summary.csv")
        with open(summary_path, "w", newline="") as fh:
            fieldnames = list(comparison_summary_rows[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_summary_rows)
        log.info("Group comparison summary: %s", summary_path)

    log.info("Group PLV analysis complete.")


# ===========================================================================
# 11. CLI
# ===========================================================================

def _write_runtime_report(cfg: Config, out_dir: str,
                          mode: str, groups: Sequence[str],
                          n_files: int) -> None:
    """Dump a JSON snapshot of the configuration used for this run."""
    report = {
        "pipeline":    "plv_pipeline.py",
        "version":     "1.0.0",
        "mode":        mode,
        "groups":      list(groups),
        "n_files":     n_files,
        "config":      {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in asdict(cfg).items()},
    }
    with open(os.path.join(out_dir, "run_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="PLV analysis pipeline for the DMN JoVE dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=("individual", "group", "interactive"),
                    default="interactive",
                    help="Analysis mode. 'interactive' shows a text menu.")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="One or two group labels for --mode group "
                         "(e.g. BL MDD or BL-ctrl).")
    ap.add_argument("--data-dir", default=None,
                    help="Directory containing *_raw-tfr-complex.h5 files. "
                         "Default: this script's directory.")
    ap.add_argument("--output-dir", default=None,
                    help="Parent directory for Results/. Default: script dir.")
    ap.add_argument("--fdr-alpha", type=float, default=None,
                    help="Override BH-FDR q-threshold.")
    ap.add_argument("--no-edge-buffer", action="store_true",
                    help="Disable the 2-second wavelet COI edge buffer.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="DEBUG-level logging.")
    return ap


def _interactive_menu(cfg: Config) -> None:
    """Tiny text menu for users who run the script by double-clicking."""
    print("\n== PLV Pipeline -- Interactive Menu ==\n")
    print("  1. Individual-subject PLV analysis (all subjects, all bands)")
    print("  2. Group-level mean PLV (one group)")
    print("  3. Between-group PLV comparison (two groups)\n")
    choice = input("Enter 1 / 2 / 3 (default 1): ").strip() or "1"
    if choice == "1":
        run_individual_pipeline(cfg)
        return
    if choice in ("2", "3"):
        n = 1 if choice == "2" else 2
        print(f"\nKnown group labels: BL, MDD, LSD, BL-ctrl, MDD-ctrl, LSD-ctrl")
        groups_raw = input(f"Enter {n} group label(s), comma-separated: ").strip()
        groups = [g.strip() for g in groups_raw.split(",") if g.strip()]
        if len(groups) != n:
            print(f"Expected {n} group label(s); got {len(groups)}. Aborting.")
            return
        run_group_pipeline(cfg, groups)
        return
    print("Unknown choice. Aborting.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config()
    if args.data_dir   is not None: cfg.data_dir   = args.data_dir
    if args.output_dir is not None: cfg.output_dir = args.output_dir
    if args.fdr_alpha  is not None: cfg.fdr_alpha  = args.fdr_alpha
    if args.no_edge_buffer:         cfg.edge_buffer_s = 0.0

    data_dir, out_dir = resolve_paths(cfg)
    log.info("Data directory:   %s", data_dir)
    log.info("Output directory: %s", out_dir)

    files = discover_complex_tfr_files(data_dir)
    log.info("%d *_raw-tfr-complex.h5 file(s) found.", len(files))

    if args.mode == "interactive":
        _interactive_menu(cfg)
    elif args.mode == "individual":
        run_individual_pipeline(cfg, files)
        _write_runtime_report(cfg, out_dir, "individual", [], len(files))
    elif args.mode == "group":
        if not args.groups:
            log.error("--groups is required with --mode group.")
            return 2
        run_group_pipeline(cfg, args.groups, files)
        _write_runtime_report(cfg, out_dir, "group", args.groups, len(files))
    else:
        log.error("Unknown mode %s", args.mode)
        return 2

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
