# PLV pipeline — murine DMN electrophysiology

A single-file,Python implementation of **Phase-Locking Value
(PLV)** and **Pairwise Phase Consistency (PPC)** analysis for complex-wavelet
LFP/ECoG recordings. This program and repository is accompanying Journal of Visualized Experiments (JoVE) manuscript (Castrén
Laboratory, University of Helsinki).

---

## 1. Scientific summary

For each pair of anatomical regions *(i, j)* and each frequency band *b*, the
pipeline reduces the `(n_channels, n_frequencies, n_timepoints)` complex
Morlet-wavelet tensor to a region × region phase-synchrony matrix.

**Phase-Locking Value** (Lachaux et al. 1999):

    PLV_ij = | mean_t( exp( i · (φ_i(t) − φ_j(t)) ) ) |

where `φ_i(t)` is the instantaneous phase extracted from the unit-normalised
complex wavelet coefficient. PLV is bounded in [0, 1].

**Pairwise Phase Consistency** (Vinck et al. 2010):

    PPC_ij = ( T · PLV_ij² − 1 ) / ( T − 1 )

PPC is an unbiased estimator of the squared population PLV; unlike PLV itself,
its expectation under the null of independent phases is exactly zero, which
removes the 1/√T positive bias that corrupts small-sample PLV comparisons.

Internally, PLV is computed in one pass per region pair via the vectorised
unit-phase outer product

    U_i(t) = exp( i · φ_i(t) )
    PLV_ij = | ( U_i · U_j^* )_t ⁄ T |

which is numerically identical to the per-pair formulation but two orders of
magnitude faster than an explicit Python loop, because the concatenated
`(n_regions, n_time)` unit-phase matrix is multiplied in a single BLAS call
(`U @ U.conj().T / T`).

Frequency-band matrices are obtained by averaging phase-synchrony across all
wavelet frequencies that fall inside the band boundaries. Channels within the
same anatomical region are first collapsed into a per-region mean wavelet
coefficient.

### Statistical inference (group mode)

For each ordered pair of groups *(A, B)* defined on the command line
(e.g. `--groups BL MDD`), every off-diagonal edge is tested with an
**independent-samples Welch t-test** (unequal variances, two-sided). The
upper-triangle family of edge p-values is corrected for multiple comparisons
using the Benjamini–Hochberg FDR procedure (Benjamini & Hochberg 1995) at
`α = 0.05` by default. Per-edge effect sizes are reported as pooled-SD Cohen's
*d*. No other inference is performed.

### Scope and conventions

* **Frequency bands (Hz)**: Delta 1–4, Theta 4–8, Alpha 8–12, Beta 12–30,
  LowGamma 30–60, HighGamma 60–100.
* **Default wavelet grid**: 35 log-spaced frequencies from ~3 to 240 Hz.
* **Sampling rate**: 250 Hz (hard-assumed; the loader does not resample).
* **Epoch**: 0–120 s by default, with a 2-s edge buffer on each side to
  mitigate wavelet cone-of-influence artefacts. If a recording is shorter than
  `2 × edge_buffer`, the buffer is silently disabled and a warning printed.
* **Regions**: 4-node eCoG DMN (R-ACA, L-ACA, R-RSP, L-RSP), ProbeA
  hippocampal/thalamic/visual column, ProbeB mPFC column. The exact
  channel-to-region mappings are embedded in `plv_pipeline.py` and taken from
  the current lab mapping as of April 2026.

---

## 2. Installation

### Prerequisites

* Python ≥ 3.9 (tested on 3.10 and 3.12)
* ~1 GB free RAM per subject for 120-s, 384-channel, 35-frequency recordings
* Linux / macOS / Windows (WSL recommended on Windows)

### Recommended: isolated virtual environment

```bash
git clone https://github.com/RazBalin/Jove_plv_pipeline.git
cd plv_pipeline

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies (pinned loosely by major version)

| Package      | Minimum    | Purpose                                    |
|--------------|------------|--------------------------------------------|
| numpy        | ≥ 1.23     | Core tensor algebra, BLAS-backed PLV       |
| scipy        | ≥ 1.10     | Welch's t-test                             |
| h5py         | ≥ 3.7      | Load `_raw-tfr-complex.h5` files           |
| matplotlib   | ≥ 3.6      | PNG output of region × region matrices    |
| statsmodels  | ≥ 0.14     | Benjamini–Hochberg FDR                     |
| pytest       | ≥ 7.0      | Optional, to run the correctness tests     |

See `requirements.txt` for the exact pinning strategy.

### Verification

After installation, run the unit tests. They take a few seconds and should
report six passes:

```bash
python -m pytest tests/ -v
```

The tests construct synthetic coupled oscillators and assert, among other
things, that (a) two phase-locked 10 Hz tones give PLV ≈ 1, (b) pure
Gaussian noise gives PLV ≈ 0 and PPC ≈ 0, (c) the FDR mask is symmetric, and
(d) the full HDF5 → region-dict → PLV-matrix round-trip succeeds.

---

## 3. Data format

The pipeline expects one HDF5 file per subject, per probe, per experimental
condition, following the Castrén lab filename convention:

```
{AnimalID}_{Probe}_{Group}_raw-tfr-complex.h5
```

Examples:

```
rb10_ProbeA_BL_raw-tfr-complex.h5         # Baseline, experiment group
rb10_ProbeB_MDD-ctrl_raw-tfr-complex.h5   # Post-stress, control group
rb12_FPGA_LSD_raw-tfr-complex.h5          # Post-LSD, experiment group
```

* `AnimalID` matches the regex `rb[a-z0-9]+`.
* `Probe` is `ProbeA`, `ProbeB`, or `FPGA` (eCoG). The token `ecog` is
  accepted as a synonym of `FPGA`.
* `Group` is one of `BL`, `MDD`, `LSD` (experiment) or their
  `-ctrl` variants (control). See the group table in the main manuscript.

Each file must contain a dataset named exactly `wavelet_complex` with shape
`(n_channels, n_frequencies, n_timepoints)` and dtype `complex128` (or
compatible). `n_channels` is 384 for Neuropixels probes (ProbeA/ProbeB) and
typically varies for FPGA; the loader uses the region mappings hard-coded in
`plv_pipeline.py` to gate only the channels that participate in a defined
anatomical region.

A minimal dummy file suitable for smoke-testing can be constructed as follows:

```python
import h5py, numpy as np
n_ch, n_f, n_t = 384, 35, 30_000            # 120 s @ 250 Hz
wav = np.zeros((n_ch, n_f, n_t), dtype=np.complex128)
# ... populate wav with your wavelet coefficients ...
with h5py.File("rb99_ProbeB_BL_raw-tfr-complex.h5", "w") as f:
    f.create_dataset("wavelet_complex", data=wav,
                     compression="gzip", compression_opts=3)
```

---

## 4. Usage

The pipeline is a single Python file with a unified command-line interface.
Three modes of operation are supported.

### Mode 1 — individual (per-subject)

Produces a region × region PLV (and PPC) matrix per subject, per file, per
band. Use this when you want to inspect one animal at a time, or when you
want the raw per-subject matrices to feed a separate statistical workflow.

```bash
python plv_pipeline.py --mode individual \
    --data-dir /path/to/tfr_complex_files \
    --output-dir /path/to/Results
```

### Mode 2 — group (between-group inference)

Loads all matching files in `--data-dir`, partitions them by the
`Group/Timepoint` token in the filename, computes subject-level matrices, then
runs an edge-wise Welch t-test with BH-FDR across the upper-triangle edge
family. Exactly two group labels must be provided.

```bash
python plv_pipeline.py --mode group \
    --groups BL MDD \
    --data-dir /path/to/tfr_complex_files \
    --output-dir /path/to/Results \
    --fdr-alpha 0.05
```

Typical group pairings that correspond to the scientific questions in the
manuscript:

| Question                        | Groups            |
|---------------------------------|-------------------|
| Stress effect (experiment arm)  | `BL MDD`          |
| Stress-group baseline equivalence | `BL BL-ctrl`    |
| LSD rescue (experiment arm)     | `MDD LSD`         |
| LSD specificity                 | `LSD LSD-ctrl`    |

### Mode 3 — interactive

A minimal menu-driven wrapper for exploratory use on a single machine; not
intended for batch / CI workflows.

```bash
python plv_pipeline.py --mode interactive
```

### Command-line options

```
--mode {individual,group,interactive}  Pipeline mode (required).
--data-dir PATH                        Directory containing *_raw-tfr-complex.h5 files.
--output-dir PATH                      Parent directory for the Results/ subfolder.
--groups A B                           Two group tags for --mode group (required there).
--fdr-alpha FLOAT                      BH-FDR α (default 0.05).
--no-edge-buffer                       Disable the 2-s COI buffer (mostly for debugging).
--use-plv                              Report raw PLV instead of the bias-corrected PPC
                                       as the primary metric. PPC is the default because
                                       it is unbiased under the null.
-v, --verbose                          Verbose logging to stderr.
```

All input / output paths are resolved relative to the current working
directory unless absolute. The pipeline does not auto-discover files
recursively: drop every HDF5 file you want analysed into the one flat
`--data-dir`.

---

## 5. Output layout

```
Results/
├── individual/
│   ├── rb10_ProbeA_BL_PLV_Alpha.npz       # matrix + region list + metadata
│   ├── rb10_ProbeA_BL_PLV_Alpha.csv       # tidy edge-list CSV
│   ├── rb10_ProbeA_BL_PLV_Alpha.png       # heatmap, 300 DPI
│   ├── ... (one triplet per subject × band)
│   └── individual_plv_summary.csv         # one row per subject × band × edge
└── group/
    ├── BL_vs_MDD_ProbeA_Alpha_mean.npz    # group-mean matrices
    ├── BL_vs_MDD_ProbeA_Alpha_mean.png
    ├── BL_vs_MDD_ProbeA_Alpha_tstat.npz
    ├── BL_vs_MDD_ProbeA_Alpha_fdr.npz
    ├── BL_vs_MDD_ProbeA_Alpha.png         # significance-masked heatmap
    ├── group_comparison_summary.csv       # one row per probe × band × edge
    └── run_report.json                    # CLI args, version, timestamps
```

The `.npz` archives always contain:

| Key            | Dtype        | Description                                  |
|----------------|--------------|----------------------------------------------|
| `matrix`       | float64      | Region × region PLV (or PPC) or statistic    |
| `regions`      | str (object) | Ordered region labels                        |
| `band`         | str          | Band name (e.g. `"Alpha"`)                   |
| `probe`        | str          | `ProbeA` / `ProbeB` / `FPGA`                 |
| `n_subjects`   | int          | (group mode only)                            |

The tidy CSVs (one row per edge) are the recommended entry point for
downstream plotting or statistical re-analysis, and are the format we
recommend peer reviewers inspect.

---

## 6. Reproducing the manuscript figure

To reproduce **"Control group PLV after 21 days (α-band)"** from the JoVE
paper, place every `rbNN_FPGA_BL-ctrl_raw-tfr-complex.h5` file in a folder
and run:

```bash
python plv_pipeline.py --mode group \
    --groups BL-ctrl MDD-ctrl \
    --data-dir /path/to/BL_ctrl_and_MDD_ctrl_h5 \
    --output-dir .
```

The file `Results/group/BL-ctrl_vs_MDD-ctrl_FPGA_Alpha_mean.png` is the
panel in the article (restricted to the BL-ctrl arm).

---

## 7. Correctness testing

The file `tests/test_plv_synthetic.py` contains six tests:

1. **Phase-locked pair → PLV ≈ 1.** Two 10 Hz oscillators with a fixed π/4
   phase offset must yield PLV > 0.9 in the alpha band.
2. **Pure noise → PPC ≈ 0.** Two independent Gaussian noise signals must
   yield PLV < 0.05 and |PPC| < 0.01 — this test specifically catches a
   missing bias correction.
3. **HDF5 round-trip.** A synthetic 384-channel `wavelet_complex` dataset
   written to disk in the production layout must be recovered by
   `load_complex_tfr` with the expected regions present.
4. **Filename parsing.** `identify_group()` and `identify_probe()` must
   return the right tags for every filename pattern used in the lab.
5. **Edge-buffer robustness.** An edge buffer longer than the epoch must not
   crash the pipeline; it must fall back to the full signal with a warning.
6. **FDR correction correctness.** The edge-wise Welch + BH-FDR pipeline
   must produce a symmetric reject mask and reject at least half of the
   strongly separated synthetic edges.

Run with:

```bash
python -m pytest tests/ -v
# or, equivalently:
python tests/test_plv_synthetic.py
```

All six tests must pass on every supported Python / NumPy combination before
merging changes.

---

## 8. Method citation

If you use this pipeline in an academic publication, please cite:

* **The PLV / PPC formulation**
  * Lachaux JP, Rodriguez E, Martinerie J, Varela FJ (1999). *Measuring phase
    synchrony in brain signals.* Hum Brain Mapp 8(4): 194–208.
  * Vinck M, van Wingerden M, Womelsdorf T, Fries P, Pennartz CMA (2010).
    *The pairwise phase consistency: a bias-free measure of rhythmic neuronal
    synchronization.* NeuroImage 51(1): 112–122.
* **The FDR correction**
  * Benjamini Y, Hochberg Y (1995). *Controlling the false discovery rate: a
    practical and powerful approach to multiple testing.* J R Stat Soc B
    57(1): 289–300.
* **The DMN electrophysiology study this pipeline accompanies**
  * (Full JoVE citation to be inserted after acceptance.)
  * https://www.biorxiv.org/content/10.1101/2025.11.13.688186v2

A machine-readable citation file (`CITATION.cff`) is included at the
repository root so GitHub can render the "Cite this repository" button.

---

## 9. License

Released under the MIT License. See `LICENSE`.

---

## 10. Contact

Raz B. — Castrén Laboratory, Neuroscience Center, University of Helsinki.
Correspondence about this repository should be routed through the
corresponding author of the JoVE manuscript.
