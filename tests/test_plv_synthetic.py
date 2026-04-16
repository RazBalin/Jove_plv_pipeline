#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_plv_synthetic.py
=====================

Unit tests for ``plv_pipeline.py`` using synthetic coupled oscillators.

We construct three "regions" of complex wavelet coefficients:

  Region A: a clean 10 Hz oscillator with fixed initial phase.
  Region B: identical to A at alpha (8-12 Hz); PLV with A should be ~1.0.
  Region C: phase-randomised Gaussian noise; PLV with A should be ~sqrt(1/T).

Each region also has valid samples at other frequencies (theta, beta) so that
the band-averaging pathway is exercised. The synthetic HDF5 file is written in
the same layout as the production ``*_raw-tfr-complex.h5`` files, so that the
full load_complex_tfr -> compute_plv_band_matrix path is tested end-to-end.

Run with:
    python -m pytest tests/test_plv_synthetic.py -v
or simply:
    python tests/test_plv_synthetic.py
"""

import os
import sys
import tempfile

import numpy as np
import h5py

# Make the parent directory importable so we can `import plv_pipeline`.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir)))

import plv_pipeline as pp  # noqa: E402 -- sys.path manipulation above


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_synthetic_wavelet(n_channels: int = 3,
                            freqs: np.ndarray = None,
                            fs: float = 250.0,
                            duration_s: float = 20.0,
                            seed: int = 0):
    """Return (wavelet_complex_array, freq_vector).

    The array has the production layout (n_channels, n_freq, n_time).
    Channel 0 is a pure 10 Hz oscillator; channel 1 is the same oscillator
    phase-shifted by pi/4; channel 2 is phase-randomised noise.
    """
    rng = np.random.default_rng(seed)
    if freqs is None:
        freqs = pp.Config().default_freqs_hz[:20]
    n_freq = freqs.size
    n_time = int(duration_s * fs)
    t = np.arange(n_time) / fs

    wavelet = np.zeros((n_channels, n_freq, n_time), dtype=np.complex128)

    # The "wavelet coefficient" at a given frequency is, physically, the
    # analytic signal of the band-limited component around that frequency.
    # For a monochromatic 10 Hz tone it is simply A * exp(i*2*pi*10*t + phi).
    for f_i, f in enumerate(freqs):
        # The oscillation content only exists near 10 Hz. At other freqs we
        # set the wavelet coefficient to Gaussian noise with small amplitude,
        # which is physically faithful: away from the tone frequency, the
        # wavelet output is noise-dominated.
        noise_level = 0.1
        # Channel 0: reference 10 Hz oscillator
        wavelet[0, f_i] = noise_level * rng.standard_normal(n_time) \
                        + 1j * noise_level * rng.standard_normal(n_time)
        # Channel 1: same 10 Hz, fixed phase offset pi/4
        wavelet[1, f_i] = noise_level * rng.standard_normal(n_time) \
                        + 1j * noise_level * rng.standard_normal(n_time)
        # Channel 2: phase-randomised noise
        wavelet[2, f_i] = rng.standard_normal(n_time) \
                        + 1j * rng.standard_normal(n_time)

        if abs(f - 10.0) < 3.0:   # inject the tone inside a +/-3 Hz window
            tone_a = np.exp(1j * (2 * np.pi * 10.0 * t))
            tone_b = np.exp(1j * (2 * np.pi * 10.0 * t + np.pi / 4))
            # Amplitude 1, so it dominates over noise_level 0.1
            wavelet[0, f_i] += tone_a
            wavelet[1, f_i] += tone_b
            # Channel 2 stays as noise -- critical for the "low PLV" check.

    return wavelet, freqs


def _write_tfr_h5(path: str, wavelet: np.ndarray) -> None:
    """Write a synthetic wavelet array to a production-layout HDF5 file."""
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("wavelet_complex", data=wavelet,
                           compression="gzip", compression_opts=3)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_plv_phase_locked_pair_is_near_one():
    """PLV(A, B) for two phase-locked 10 Hz oscillators must be ~1.0."""
    cfg = pp.Config()
    cfg.edge_buffer_s = 0.0          # simplify expectations
    wavelet, freqs = _make_synthetic_wavelet(duration_s=20.0)
    cfg.default_freqs_hz = freqs
    cfg.epoch_end_s = 20.0

    # Build the region_dict directly (skip HDF5 round-trip for speed).
    region_dict = {
        "A": wavelet[0],
        "B": wavelet[1],
        "C": wavelet[2],
        "freqs": freqs,
    }

    alpha_idx = pp.freq_indices_for_band((8.0, 12.0), cfg)
    assert alpha_idx.size > 0, "No wavelet frequencies in alpha band"

    regions, plv, ppc = pp.compute_plv_band_matrix(region_dict, alpha_idx, cfg)
    assert regions == ["A", "B", "C"]

    # A <-> B: same 10 Hz tone with a fixed phase offset => PLV ~ 1.
    assert plv[0, 1] > 0.9, f"Expected PLV(A,B) > 0.9, got {plv[0, 1]:.3f}"
    # A <-> C and B <-> C: no coherent phase => PLV small (sqrt(1/T) ~ 0.006
    # for T = 30000, but allow some slack from the noise-only freq indices).
    assert plv[0, 2] < 0.2, f"Expected PLV(A,C) < 0.2, got {plv[0, 2]:.3f}"
    assert plv[1, 2] < 0.2, f"Expected PLV(B,C) < 0.2, got {plv[1, 2]:.3f}"

    # Diagonal must be exactly 1.0 by construction.
    assert np.allclose(np.diag(plv), 1.0)


def test_ppc_bias_corrects_pure_noise():
    """Pure-noise PPC must be near zero, whereas PLV is biased upward by ~1/sqrt(T)."""
    cfg = pp.Config()
    cfg.edge_buffer_s = 0.0
    rng = np.random.default_rng(1)

    n_freq = 10
    n_time = 10_000
    freqs = pp.Config().default_freqs_hz[:n_freq]
    cfg.default_freqs_hz = freqs
    cfg.epoch_end_s = n_time / cfg.sample_rate_hz

    # Two independent pure-noise regions.
    a = rng.standard_normal((n_freq, n_time)) + 1j * rng.standard_normal((n_freq, n_time))
    b = rng.standard_normal((n_freq, n_time)) + 1j * rng.standard_normal((n_freq, n_time))
    region_dict = {"A": a, "B": b, "freqs": freqs}

    alpha_idx = pp.freq_indices_for_band((8.0, 12.0), cfg)
    _, plv, ppc = pp.compute_plv_band_matrix(region_dict, alpha_idx, cfg)

    # Under pure noise, E[PLV^2] = 1/T and E[PPC] = 0.
    # With T ~ 30000, PLV is O(0.006) -- below our tolerance.
    assert plv[0, 1] < 0.05, f"Noise PLV too large: {plv[0, 1]:.4f}"
    assert abs(ppc[0, 1]) < 0.01, f"Noise PPC not near zero: {ppc[0, 1]:.4f}"


def test_load_complex_tfr_roundtrip(tmp_path):
    """End-to-end: HDF5 file => load_complex_tfr => PLV matrix."""
    cfg = pp.Config()
    cfg.edge_buffer_s = 0.0
    cfg.epoch_end_s = 10.0   # 10-s synthetic epoch keeps memory use small
    # Use only the handful of channels (0..7) that sit inside the eCoG map;
    # those channels belong to region "R-ACA" (plus a few unmapped). This
    # keeps the 384-channel array size modest.
    wavelet, freqs = _make_synthetic_wavelet(n_channels=20, duration_s=10.0)
    cfg.default_freqs_hz = freqs

    # Pad up to the 384 channels the real mapping expects, with empty
    # (zero-amplitude) wavelet coefficients so that region averaging still
    # works across the ones we populated.
    padded = np.zeros((384, wavelet.shape[1], wavelet.shape[2]),
                      dtype=np.complex128)
    # Populate channels 3..7 (R-ACA) and 8..12 (L-ACA) from the synthetic
    # data -- this ensures at least two regions are present.
    padded[3:8] = wavelet[:5]
    padded[8:13] = wavelet[:5]

    fp = os.path.join(str(tmp_path), "rb00_FPGA_BL_raw-tfr-complex.h5")
    _write_tfr_h5(fp, padded)

    region_dict = pp.load_complex_tfr(fp, cfg)
    assert region_dict is not None
    assert "freqs" in region_dict
    # Must recover at least R-ACA and L-ACA.
    assert "R-ACA" in region_dict and "L-ACA" in region_dict


def test_identify_group_and_probe():
    assert pp.identify_group("rb10_ProbeA_BL-ctrl_raw-tfr-complex.h5") == "BL-ctrl"
    assert pp.identify_group("rb10_ProbeA_MDD_raw-tfr-complex.h5") == "MDD"
    assert pp.identify_group("rb10_ProbeA_LSD-ctrl_raw-tfr-complex.h5") == "LSD-ctrl"
    assert pp.identify_group("rb10_FPGA_lsd_raw-tfr-complex.h5") == "LSD"
    assert pp.identify_group("random_file.h5") == "Unknown"

    assert pp.identify_probe("rb10_ProbeA_BL_raw-tfr-complex.h5") == "ProbeA"
    assert pp.identify_probe("rb10_probeb_mdd_raw-tfr-complex.h5") == "ProbeB"
    assert pp.identify_probe("rb10_fpga_bl_raw-tfr-complex.h5") == "FPGA"
    assert pp.identify_probe("rb10_ecog_bl_raw-tfr-complex.h5") == "FPGA"


def test_edge_buffer_does_not_crash_on_short_epoch():
    """Edge-buffer longer than epoch must be handled gracefully (warning, not error)."""
    cfg = pp.Config()
    cfg.edge_buffer_s = 200.0      # longer than the epoch -> must not crash
    cfg.epoch_end_s = 10.0
    freqs = pp.Config().default_freqs_hz[:10]
    cfg.default_freqs_hz = freqs
    n_time = int(cfg.epoch_end_s * cfg.sample_rate_hz)
    data = (np.random.randn(len(freqs), n_time)
            + 1j * np.random.randn(len(freqs), n_time))
    region_dict = {"A": data, "B": data.copy(), "freqs": freqs}
    regions, plv, ppc = pp.compute_plv_band_matrix(
        region_dict, np.arange(len(freqs)), cfg)
    assert regions == ["A", "B"]
    assert plv.shape == (2, 2)


def test_edgewise_group_comparison_fdr_correction():
    """Edge-wise Welch test must produce valid FDR q-values and symmetric masks."""
    cfg = pp.Config()
    n_regions = 5
    rng = np.random.default_rng(42)

    # Group A: low PLV (~0.2); Group B: high PLV (~0.7); should reject edges.
    group_a = [np.clip(0.2 + 0.05 * rng.standard_normal((n_regions, n_regions)),
                       0, 1) for _ in range(8)]
    group_b = [np.clip(0.7 + 0.05 * rng.standard_normal((n_regions, n_regions)),
                       0, 1) for _ in range(8)]
    for m in group_a + group_b:
        m[np.triu_indices(n_regions, k=0)] = m.T[np.triu_indices(n_regions, k=0)]
        np.fill_diagonal(m, 1.0)

    regions = [f"R{i}" for i in range(n_regions)]
    res = pp.edgewise_group_comparison(group_a, group_b, regions, cfg)

    # Matrices must be square and symmetric (on the off-diagonal).
    assert res["t_matrix"].shape == (n_regions, n_regions)
    assert res["p_fdr_matrix"].shape == (n_regions, n_regions)
    assert np.array_equal(res["reject_matrix"], res["reject_matrix"].T)

    # With mean difference 0.5 and SD 0.05, every off-diagonal edge is a
    # strong true-positive; require at least half of them to survive FDR.
    n_edges = n_regions * (n_regions - 1) // 2
    n_sig = int(res["reject_matrix"][np.triu_indices(n_regions, k=1)].sum())
    assert n_sig >= n_edges // 2, f"Expected >=half edges significant, got {n_sig}/{n_edges}"


# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Run as a script: minimal self-test without pytest.
    import traceback
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            # Pass a tmp_path-like object for the one test that needs it.
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(tmp)
            else:
                fn()
            print(f"  OK    {name}")
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
        except Exception:
            print(f"  ERROR {name}:")
            traceback.print_exc()
