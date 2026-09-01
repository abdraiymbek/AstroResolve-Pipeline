# Astronomical Super-Resolution Research

## 1. Project identity

**Working title:** Astronomical Super-Resolution Research: DLSS-Inspired Recursive Cosmic Zoom

**Project type:** Reproducible research prototype, not a production observatory pipeline

**Primary implementation language:** Python

**Likely libraries:** PyTorch, torchvision, OpenCV, Astropy, NumPy, SciPy, scikit-image, pandas, matplotlib, seaborn, PyYAML, and a metrics library where useful. Prefer permissively licensed, well-maintained open-source components.

**Initial software budget:** As close to zero as practical. Use public data, local CPU or an existing GPU, open-source models, and open-source software. Do not purchase cloud compute or proprietary APIs during P0 without an explicit decision recorded in the experiment log.

**Research status:** Exploratory. No result, model, or claim is established in advance.

**Scientific owner:** The project author. A later review/mentorship discussion is planned with Dr. Kevin Crosby after an initial evidence package exists.

**Publication intent:** If the results are scientifically meaningful and reproducible, prepare a research paper with methods, limitations, failure cases, and reproducibility artifacts.

This document is the source of truth for the coding agent working on the project. The agent must preserve the scientific intent, the limitations, and the falsifiability of the research even when implementation details change.

---

## 2. Executive summary

The project investigates whether a DLSS-inspired, stochastic, recursive super-resolution procedure can recover *reliable, repeatable structure* from undersampled or degraded astronomical observations. The central idea is to apply an image super-resolution model many times with controlled stochastic variation, identify structures that persist across independent reconstructions, quantify disagreement and uncertainty, and permit another scale-up only when measurable confidence gates pass.

The concept is sometimes described informally as a recursive “cosmic zoom.” That phrase is useful for communicating the workflow, but it must not be interpreted literally. The system cannot create information absent from the measurements. Telescope optics, the point-spread function (PSF), pixel sampling, diffraction, detector noise, sky/background noise, saturation, compression, and the observing setup impose physical information limits. A visually sharp or model-consistent feature is not automatically a real astronomical feature.

The first objective is therefore not to train a new astronomy model. It is to build and test the complete algorithm using existing pretrained super-resolution baselines and physically motivated synthetic degradations. The project must establish evidence before investing in custom training or substantial GPU compute.

The intended scientific claim, if supported, is narrow:

> Under specified observing and degradation conditions, repeated stochastic reconstruction plus cross-run consensus may improve the recovery and measurement of structures that are supported by the data, while providing useful indicators of instability for unsupported structures.

The project must not claim that arbitrary detail beyond the telescope’s information content has been recovered, that consensus proves truth, or that a large enlargement factor is scientifically valid merely because the output looks plausible.

---

## 3. Core research question

### 3.1 Primary question

Can repeated stochastic super-resolution reconstructions, combined with cross-run consensus, uncertainty estimation, and gated recursion, recover astronomically meaningful structure more reliably than standard interpolation, classical restoration, or a single deterministic/pretrained super-resolution model when evaluated against known high-resolution ground truth and realistic degraded observations?

### 3.2 Secondary questions

1. Which kinds of structure are stable across stochastic reconstructions: point sources, extended morphology, edges, filaments, rings, texture, flux distributions, or only low-frequency appearance?
2. Does consensus reduce hallucinated or unstable detail, or does it merely produce a smoother and more confident-looking average?
3. How does performance change as PSF width, sampling, signal-to-noise ratio, background, detector noise, and degradation mismatch become more severe?
4. How many stochastic samples are useful before additional samples provide negligible benefit relative to compute cost?
5. Under what measurable conditions does recursive scaling help, become neutral, or amplify error?
6. Are image-level improvements accompanied by better photometric, astrometric, and morphological fidelity?
7. Can uncertainty or disagreement maps identify regions where the model should be trusted less?

---

## 4. Hypotheses and falsifiability

The hypotheses are deliberately testable and may be rejected.

### H1 — Consensus stability hypothesis

Structures supported by the observation will show greater cross-run stability than unsupported model-invented structures. On held-out data, consensus/stability scores should correlate with reconstruction fidelity better than a raw visual inspection or a single model confidence score.

**Falsifier:** Stability is uncorrelated with truth, or consistently high stability is observed for incorrect structures.

### H2 — Uncertainty usefulness hypothesis

Ensemble disagreement and related uncertainty maps will identify regions, objects, or measurements with higher reconstruction error and will enable a selective output mode that improves reliability by abstaining on low-confidence regions.

**Falsifier:** Uncertainty is poorly calibrated, does not rank errors, or selective abstention does not improve the retained set’s fidelity.

### H3 — Gated recursion hypothesis

Recursive scale-up will improve useful resolution only when the preceding reconstruction meets validation gates. Unconditionally repeated scaling will perform worse or produce more unsupported structure than gated recursion.

**Falsifier:** Recursion provides no benefit over one pass, or the gate does not distinguish beneficial from harmful recursion.

### H4 — Domain and degradation hypothesis

Performance depends strongly on whether the model and synthetic degradation resemble the astronomical domain and instrument. A generic pretrained model may improve perceptual appearance while harming scientific measurements, especially under out-of-domain conditions.

**Falsifier:** Domain mismatch and degradation mismatch have no measurable effect, or perceptual improvements reliably imply scientific fidelity across tests.

### H5 — Sample-efficiency hypothesis

The benefit of increasing the ensemble size will show diminishing returns. A configurable, experimentally justified sample count will reach a useful stability/compute trade-off before the historical upper bound of approximately 1,000 samples per scale in the most expensive experiments.

**Falsifier:** Results remain materially unstable at the chosen maximum, or very small ensembles perform identically under all conditions.

The project must report negative, mixed, and inconclusive results. Do not adjust thresholds after inspecting held-out test outcomes unless the change is logged as a new experiment.

---

## 5. Scientific boundaries and anti-hallucination policy

These are hard requirements, not optional wording.

1. **No information-from-nothing claim.** The system may estimate a likely high-resolution representation, but it cannot recover detail that has been destroyed by sampling, diffraction, the PSF, noise, saturation, or other information loss.
2. **Consensus is not proof.** Agreement across model runs is evidence of model stability under the chosen perturbations. It is not independent observational evidence and cannot establish that a feature is real.
3. **Confidence must be earned.** Until calibrated against held-out truth, use terms such as *stability*, *agreement*, or *empirical confidence proxy* rather than implying a formal probability of truth.
4. **Abstention is a valid result.** The pipeline must be able to reject a scale-up, mark a region as unreliable, and return the best validated lower-resolution result.
5. **Preserve the observation.** Every output must retain access to the original measurement, the forward-model assumptions, and a mapping back to the input. The enhanced image must never silently replace the observation.
6. **No cherry-picking.** Dataset partitions, random seeds, sample counts, thresholds, examples, and failed cases must be logged before or at the time of execution. Do not select visually attractive examples as representative after the fact.
7. **No hidden post-processing.** Any denoising, deconvolution, sharpening, clipping, color transform, normalization, or display stretch must be explicit and separately recorded from the scientific reconstruction.
8. **No false precision.** Do not report more photometric, astrometric, or morphological precision than supported by the input and validation uncertainty.
9. **No leakage.** Images, crops, tiles, objects, fields, augmentations, or near-duplicates from a test source must not appear in training, tuning, threshold selection, or prompt/model adaptation.
10. **No unsupported scientific interpretation.** A reconstructed feature must not be described as a physical object, event, or structure without independent evidence and expert review.

Every report should include a short limitations section using this language or an equivalent statement: the method reconstructs or infers a plausible representation under specified assumptions; it does not bypass physical information limits.

---

## 6. Scope and staged objectives

### 6.1 P0 / MVP definition

P0 is the smallest complete, scientifically testable implementation. It is complete only when all of the following work end to end on a small, documented dataset:

- Ingest or generate a small set of high-quality astronomical-like reference images.
- Apply a configurable, physically motivated degradation pipeline to produce low-resolution observations and retain the known reference.
- Run at least one interpolation baseline, one classical restoration baseline, and one existing pretrained SR model compatible with the available environment. If no compatible public checkpoint can be run, P0 is not complete; document the blocker and resolve it before drawing comparative conclusions.
- Produce multiple stochastic reconstructions from the same degraded observation using recorded seeds and perturbations.
- Compute a consensus reconstruction and a spatial disagreement/stability map.
- Apply at least one explicit recursive scale-up gate and demonstrate both pass and fail behavior on controlled examples.
- Compare outputs to the held-out reference with image-level and astronomy-relevant metrics.
- Save configurations, seeds, software metadata, data manifests, intermediate artifacts, metrics, plots, and failure cases.
- Export a concise run record suitable for eLabFTW logging.
- Run from a documented CLI command on a clean environment or documented local setup.

P0 is not required to include custom model training, a large survey, real-time performance, a web interface, a publication-ready sample size, or a claim of state-of-the-art performance.

### 6.2 Post-P0 objectives

After P0 evidence is reviewed:

1. Expand the validation set and include real instrument data where reliable paired or proxy ground truth is available.
2. Tune or replace the degradation model using instrument-specific metadata.
3. Test astronomy-specific pretrained or fine-tuned models.
4. Train a custom astronomy-specific SR model only if P0 shows a meaningful, reproducible research signal and the data/compute justification is documented.
5. Re-run the full validation and ablation suite after any custom training.
6. Seek review and mentorship from Dr. Kevin Crosby or another qualified astronomer/instrumentation expert.
7. Prepare a paper with methods, limitations, reproducibility artifacts, and negative results.

### 6.3 Explicit non-goals

- Building an autonomous telescope-control or discovery system.
- Replacing calibrated astronomical reduction pipelines.
- Claiming arbitrary zoom factors or direct visualization of unseen detail.
- Optimizing only for aesthetically pleasing images.
- Treating a single pretrained model’s output as scientific ground truth.
- Spending materially on compute before the baseline experiment justifies it.

---

## 7. Operational concept and end-to-end algorithm

The implementation should separate *measurement*, *degradation assumptions*, *model reconstruction*, *consensus*, *validation*, and *display*. A typical run is:

1. **Register a dataset and run.** Create a unique run ID. Record source, license, instrument or synthetic scene provenance, commit/version, environment, configuration hash, and seeds.
2. **Prepare references.** Convert images to a documented numeric representation. Preserve raw values and metadata. Avoid irreversible clipping. Create train/validation/test partitions by source/object/field as appropriate.
3. **Generate or ingest an observation.** For synthetic validation, start with a known high-resolution reference (x), apply the configured forward model, and obtain a low-resolution measurement (y). For real data, store the measured image and the best available metadata without pretending a reference is known.
4. **Run baselines.** Produce bicubic or Lanczos interpolation, a classical restoration output, and one or more existing SR model outputs using the same input and scale target.
5. **Generate a stochastic ensemble.** Create (K) reconstructions from the same observation, with all sources of stochasticity documented. Candidate sources include model sampling if natively supported, controlled noise/degradation perturbations, test-time augmentation, dropout only when methodologically justified, or multiple independently seeded inference paths. A deterministic model run with only a changed filename is not an ensemble.
6. **Align and normalize outputs.** Ensure all ensemble members are in the same pixel coordinate system, orientation, scale, dynamic range, and photometric convention. Record every alignment and normalization operation.
7. **Match and filter outputs.** Compare outputs at pixel, patch, feature, and object/catalog levels where meaningful. Reject gross outliers, failed inference, boundary-contaminated tiles, and outputs that violate explicit numerical checks. Use robust statistics rather than silently discarding inconvenient results.
8. **Compute consensus and uncertainty.** Produce a consensus image plus maps of dispersion, feature presence frequency, local disagreement, and any calibrated error proxy. Retain the full ensemble or a lossless reference to it for audit.
9. **Apply a recursion gate.** Permit another scale step only if predeclared gates pass. If gates fail, stop recursion and return the last accepted output with a failure reason.
10. **Validate.** On known-reference data, compare every candidate against the same held-out target. On real data without ground truth, limit claims to stability, consistency, and agreement with external measurements or higher-quality observations.
11. **Report.** Generate machine-readable metrics, plots, tables, an experiment summary, failure cases, and an eLabFTW-ready record.

### 7.1 Mathematical framing

Use a measurement model such as:

\[
y = D\big(S(h * x)\big) + n + b + a,
\]

where (x) is the latent high-resolution scene, (h) is the optical/instrument PSF, (h*x) is convolution, (S) represents sampling and pixel integration, (D) represents scale/downsampling or other instrument response, (n) is stochastic detector/read/shot noise, (b) is background or sky contribution, and (a) represents optional artifacts such as cosmic rays, bad pixels, saturation, or compression. The exact order and terms must match the chosen instrument or synthetic scenario; this equation is a conceptual starting point, not permission to use an unrealistic simulator.

The reconstruction estimates a distribution or ensemble of plausible outputs:

\[
\{\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_K\} = f_\theta(y; \omega_1), \ldots, f_\theta(y; \omega_K),
\]

where \(\omega_k\) captures a recorded stochastic path or perturbation. The consensus \(\hat{x}_{cons}\) and stability/uncertainty maps are computed from this ensemble. Any report must distinguish the observed (y), the assumed forward model, the individual \(\hat{x}_k\), and the consensus estimate.

### 7.2 Stochastic sample count

The sample count (K) must be configurable per experiment. Historically, discussions considered up to approximately 1,000 samples per scale; retain that as an upper experimental option, not a default requirement. Begin with a pilot such as 8, 16, 32, 64, and 128 samples, then extend only if a stability/metric curve justifies the cost. A sample-count study must report:

- wall-clock time and memory;
- marginal change in consensus and uncertainty maps;
- marginal change in validation metrics;
- stability of conclusions across seeds;
- whether the selected (K) changes the recursion decision.

Never use 1,000 samples merely to create an impressive number. The sample budget must be experimentally justified.

### 7.3 Consensus and match filtering

Implement consensus as a replaceable module with explicit methods. At minimum, support:

- robust pixel-wise median or trimmed mean;
- dispersion such as standard deviation, median absolute deviation, or percentile interval;
- patch/feature agreement checks after alignment;
- optional object-level matching for detected sources or segmented structures;
- configurable outlier rejection with counts and reasons;
- an option to retain all outputs and compare filtered versus unfiltered consensus.

The pipeline must test whether consensus preserves flux, centroid, morphology, and spatial frequency content. A consensus image that looks cleaner but systematically reduces peaks, merges sources, shifts centroids, or erases faint structure is not automatically an improvement.

### 7.4 Confidence and uncertainty maps

The MVP should produce spatial maps including, where defined:

- per-pixel or per-patch ensemble dispersion;
- robust dispersion or percentile intervals;
- frequency with which a feature/edge/source is detected;
- disagreement near object boundaries and high-frequency regions;
- input signal-to-noise and mask overlays;
- error calibration plots on held-out references.

The code must distinguish aleatoric uncertainty approximations, ensemble disagreement, and error measured against ground truth. Do not label any of these as a formal posterior probability unless a valid probabilistic model and calibration procedure are implemented.

### 7.5 Recursive scaling gate

The default recursion should be conservative:

- start from the measured image or a validated first-pass reconstruction;
- apply one configured scale factor, normally a modest factor such as 2x;
- generate and evaluate the ensemble;
- accept the new scale only if all required gates pass;
- stop at a configured maximum depth regardless of visual appeal.

Candidate gates include:

- minimum ensemble agreement in relevant regions;
- maximum allowed uncertainty/disagreement;
- no major increase in artifact indicators;
- acceptable photometric and astrometric consistency on synthetic validation;
- stable source count and morphology within tolerance;
- forward-projection consistency: degrading the reconstruction should remain compatible with the observation within declared tolerance;
- no saturation, clipping, NaN/Inf, or boundary failure;
- no regression beyond a predeclared tolerance relative to the prior accepted scale.

The gate must return a structured decision such as `accepted`, `rejected_uncertainty`, `rejected_forward_consistency`, `rejected_photometry`, or `max_depth_reached`. It must be possible to disable recursion for an ablation.

---

## 8. Data strategy and synthetic degradation

### 8.1 Data principles

Use data with clear provenance, license, resolution, filters/bands, instrument information, and processing history whenever available. Start with a small, manageable set for pipeline correctness, then expand for statistical conclusions. Keep raw/source references read-only and write derived data into versioned run directories.

Partitions must prevent leakage at the highest meaningful independence level. If several crops come from the same astronomical object or field, keep them in the same split. Near-duplicate images, alternate processing products, and augmented copies must not cross splits.

Maintain a manifest with at least:

- stable sample ID;
- source URL or archive identifier;
- license and attribution;
- object/field/instrument/filter metadata when known;
- original dimensions, pixel scale, units, and bit depth when known;
- preprocessing operations;
- train/validation/test assignment;
- checksum of downloaded or generated source;
- synthetic degradation configuration and seed;
- any masking or exclusion reason.

### 8.2 Synthetic forward model

The initial synthetic pipeline should expose, independently where possible:

- PSF family and width, including spatially varying PSF as a later extension;
- optical blur and optional diffraction-inspired behavior;
- pixel integration and sampling;
- downsampling factor and interpolation kernel;
- Poisson/shot noise, read noise, Gaussian noise, and background level;
- gain/exposure scaling and dynamic range;
- bad pixels, cosmic rays, saturation, missing data, and clipping;
- optional compression or resampling artifacts only when relevant;
- channel/band handling for grayscale or multiband images.

Every degraded sample must save the exact parameter values and random seed. The degradation distribution used to generate training/tuning data must be separated from the held-out test distribution. Include both in-distribution and deliberate mismatch tests.

### 8.3 Real-data track

Synthetic results are necessary but not sufficient. A separate real-data track should test whether conclusions survive real detector/systematics. If no true high-resolution ground truth exists, do not use a sharper-looking image as ground truth. Possible validation references include independent exposures, multi-frame combinations, higher-resolution instruments, known catalog positions/fluxes, or task-specific measurements, each with limitations documented.

Report synthetic and real-data results separately. Do not combine them into one score that hides the lack of ground truth.

---

## 9. Models and comparison baselines

### 9.1 Required first-stage comparisons

All methods must receive the same input, target scale, masks, dynamic-range convention, and evaluation crop policy.

At minimum compare:

1. nearest-neighbor or another simple sanity baseline where useful;
2. bicubic and/or Lanczos interpolation;
3. a classical restoration method such as Wiener filtering or Richardson–Lucy deconvolution, with parameters documented;
4. at least one existing pretrained SR model compatible with the data and hardware;
5. the proposed stochastic ensemble/consensus pipeline built around the same or comparable pretrained model.

Potential pretrained models include established generic SR architectures such as EDSR, RCAN, SwinIR, Real-ESRGAN, or another maintained implementation, subject to license, model availability, scale compatibility, and domain appropriateness. The coding agent must verify the actual package/API and license before using one; do not assume a model’s advertised perceptual quality implies astronomy fidelity.

### 9.2 Custom astronomy model

Custom training is optional and gated by evidence. Consider it only after:

- P0 runs reproducibly;
- baseline comparisons reveal a stable limitation or opportunity;
- paired/degraded training data and held-out high-quality data are available;
- a compute budget and expected benefit are written down;
- model license and data licenses permit the intended use;
- the training plan cannot leak test information.

The custom model should be trained on paired data formed from high-quality astronomical imagery and controlled degradations, with instrument/domain metadata where possible. Its evaluation must repeat the pretraining-era baseline suite and every relevant ablation. Custom training must not replace the generic baseline comparison; it extends it.

---

## 10. Evaluation and scientific metrics

No single metric is sufficient. Report per-image distributions, confidence intervals or bootstrap intervals where appropriate, and aggregate summaries by scene/object type and degradation severity.

### 10.1 Image-level metrics

- PSNR, with data-range conventions stated;
- SSIM and/or multi-scale SSIM;
- a perceptual metric only as a secondary diagnostic, not as evidence of scientific truth;
- error maps and frequency-domain diagnostics;
- input-to-output forward consistency after re-degradation.

### 10.2 Photometric fidelity

Where scientifically meaningful, measure:

- integrated flux and aperture photometry error;
- peak flux and surface-brightness error;
- flux preservation across isolated and crowded sources;
- background and noise statistics;
- magnitude or color error for multiband data;
- bias and scatter as a function of signal-to-noise and source brightness.

### 10.3 Astrometric fidelity

- source centroid error;
- positional bias and scatter;
- separation error in close pairs;
- source count and false-source rate;
- registration/alignment residuals.

### 10.4 Morphological and structural fidelity

- size, FWHM, ellipticity, orientation, and axis-ratio error;
- radial-profile agreement;
- preservation of rings, filaments, edges, knots, and other labeled structures where available;
- connected-component/segmentation agreement;
- shape or morphology classification consistency as a downstream diagnostic;
- preservation of scientifically relevant spatial frequencies without unsupported high-frequency excess.

### 10.5 Uncertainty and selective prediction

- correlation between ensemble disagreement and actual error;
- calibration curves or reliability diagrams where a probability-like score is claimed;
- coverage of prediction intervals;
- risk-coverage curves for abstention;
- fidelity among accepted versus rejected regions/objects;
- recursion-gate precision: how often accepted recursion helps versus harms.

### 10.6 Statistical discipline

Use identical evaluation code for all methods. Report variance across images and seeds. Define the primary metrics before inspecting the final test results when practical. Do not rely on one cherry-picked image, one favorable band, or one degradation setting. Include examples where every method fails.

---

## 11. Ablation plan

The minimum ablation suite should isolate what causes any observed effect:

- single model output versus stochastic ensemble;
- ensemble size (K), including a diminishing-returns curve;
- no consensus versus median/trimmed consensus/other method;
- no outlier filtering versus filtering;
- no uncertainty map versus uncertainty-aware selection;
- no recursion versus unconditional recursion versus gated recursion;
- recursion gate thresholds and individual gate components;
- model stochasticity source: sampling, augmentation, degradation perturbation, or other mechanism;
- PSF width and mismatch;
- noise level, background, and signal-to-noise;
- sampling/downsampling factor;
- ideal versus imperfect degradation knowledge;
- in-domain versus out-of-domain object/field types;
- different pretrained models;
- display-only post-processing versus scientific output;
- test-time compute budget and runtime.

Every ablation must state the controlled variables and the question it answers. If an ablation is skipped, record why.

---

## 12. Failure modes and required handling

The pipeline must detect, preserve, and report at least these cases:

- hallucinated stars, filaments, edges, texture, or sources;
- disappearance or merging of real faint structures;
- ringing, checkerboard artifacts, oversharpening, and tiling seams;
- incorrect source centroids or flux redistribution;
- unstable outputs across seeds;
- false consensus caused by shared model bias;
- confidence maps that are smooth or visually plausible but uncalibrated;
- recursive amplification of an early mistake;
- sensitivity to normalization, clipping, or display stretch;
- PSF/degradation mismatch;
- crowded-field confusion;
- saturation, cosmic rays, bad pixels, missing data, and boundary effects;
- failure on negative-valued or calibrated scientific data if the model assumes display RGB;
- nondeterminism or incompatible hardware causing unreproducible outputs.

Each failure case should include the input, the relevant output(s), ensemble disagreement, gate decision, metrics if truth exists, configuration, and a short interpretation. The correct response to a failure is often to stop, abstain, or revise the experiment—not to add unlogged post-processing.

---

## 13. Reproducibility and provenance requirements

Every experiment directory must be sufficient for another researcher to understand and, where feasible, rerun the result.

Record:

- run ID and timestamp;
- source commit or code version;
- environment/package lock and hardware information;
- dataset manifest and checksums;
- all configuration values and configuration hash;
- global and per-sample random seeds;
- model name, version, checkpoint path, checksum, and license;
- preprocessing, normalization, masks, and display settings;
- degradation parameters and seed;
- ensemble sample count and stochasticity source;
- consensus, filtering, and recursion thresholds;
- runtime, memory, failures, and skipped samples;
- metrics, plots, and report generation version;
- human decisions and deviations from the plan.

Prefer deterministic algorithms and fixed seeds where supported, but document unavoidable nondeterminism. Never overwrite a completed run. Use a new run ID for changed code, data, model, or thresholds.

---

## 14. eLabFTW research logging and export

eLabFTW is the intended research notebook/logging system. The software should not require an eLabFTW API connection for local P0 execution, but it must produce a concise export package that can be pasted or uploaded into an eLabFTW experiment entry.

The export should contain:

- experiment title and run ID;
- research question/hypothesis under test;
- date, operator, code version, and environment;
- dataset and provenance summary;
- configuration hash and key parameters;
- model/checkpoint and license;
- seed and ensemble count;
- gate decisions;
- primary results and plots;
- known failures and deviations;
- conclusion limited to the evidence;
- links or relative paths to artifacts and checksums.

If API integration is added later, make it optional, authenticated through user-provided configuration, and never include secrets in logs or exports. A local Markdown/JSON export is the P0 acceptance target.

---

## 15. Proposed repository structure

Use a structure close to the following. Adjust names only when the implementation benefits clearly and update this document if the structure changes.

```text
astronomical-super-resolution/
├── Project.MD
├── README.md
├── pyproject.toml                 # package metadata and dependencies
├── uv.lock / poetry.lock          # use one chosen lockfile
├── configs/
│   ├── p0_smoke.yaml
│   ├── synthetic_baseline.yaml
│   ├── ensemble_sample_sweep.yaml
│   ├── recursion_ablation.yaml
│   └── custom_training.yaml       # future/optional
├── src/astrsr/
│   ├── cli.py
│   ├── config.py
│   ├── provenance.py
│   ├── data/
│   │   ├── ingest.py
│   │   ├── manifest.py
│   │   ├── splits.py
│   │   └── transforms.py
│   ├── degradation/
│   │   ├── forward_model.py
│   │   ├── psf.py
│   │   ├── noise.py
│   │   └── artifacts.py
│   ├── models/
│   │   ├── registry.py
│   │   ├── baselines.py
│   │   ├── pretrained.py
│   │   └── custom.py
│   ├── ensemble/
│   │   ├── sampler.py
│   │   ├── alignment.py
│   │   ├── filtering.py
│   │   ├── consensus.py
│   │   └── uncertainty.py
│   ├── recursion/
│   │   ├── gates.py
│   │   └── runner.py
│   ├── evaluation/
│   │   ├── image_metrics.py
│   │   ├── photometry.py
│   │   ├── astrometry.py
│   │   ├── morphology.py
│   │   ├── calibration.py
│   │   └── report.py
│   ├── logging/
│   │   ├── run_record.py
│   │   └── elabftw_export.py
│   └── utils/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/                    # exploration only; canonical runs use CLI/configs
├── docs/
│   ├── methods.md
│   ├── data.md
│   └── experiment_log/
├── data/
│   ├── raw/                      # read-only or externally managed
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── checkpoints/                  # large files managed separately when needed
├── runs/                         # immutable run artifacts, not source data
└── reports/
```

Do not commit large raw datasets, private data, credentials, or unlicensed model weights. Keep source references and checksums instead where possible.

---

## 16. Module responsibilities

### Configuration and provenance

Validate schemas, apply defaults explicitly, compute a configuration hash, create run IDs, capture environment metadata, and reject unknown or misspelled parameters when possible.

### Data and manifests

Load data without silently changing units or dynamic range; create leakage-safe splits; preserve metadata; produce checksummed manifests.

### Degradation

Implement the forward measurement model as composable, testable operators. Each operator must expose parameters, seed behavior, and a human-readable description. Unit tests should verify known properties such as shape changes, noise statistics, and flux behavior where expected.

### Models

Provide a common inference interface for interpolation, classical restoration, pretrained SR, and future custom models. Keep model loading, checkpoint provenance, device selection, and license metadata separate from scientific orchestration.

### Ensemble

Run independent seeded reconstructions; retain per-sample metadata; align outputs; detect failures; compute robust consensus, dispersion, and feature/object agreement. Never silently drop an output without recording the reason.

### Recursion

Own scale scheduling, stopping conditions, gate evaluation, and decision records. It must be possible to reproduce a gate decision from saved artifacts and metrics.

### Evaluation

Compute the same metrics for every method and output. Separate image-quality metrics from scientific fidelity metrics and from visualization metrics. Support masks and uncertainty-aware aggregation.

### Reporting and eLabFTW

Generate machine-readable JSON/CSV plus human-readable Markdown/HTML/PNG outputs. Include limitations and failure cases automatically in the report template.

---

## 17. Configuration schema

Use YAML or another human-readable format validated into a typed internal object. A representative P0 configuration is:

```yaml
run:
  name: p0_smoke
  seed: 20260831
  output_dir: runs/
  device: auto
  deterministic: true

data:
  source: synthetic_fixture
  manifest: data/manifests/p0.yaml
  split: test
  channels: [intensity]
  preserve_units: true
  normalization: documented_linear

degradation:
  enabled: true
  scale_factor: 2
  psf:
    type: gaussian
    fwhm_pixels: 1.5
  sampling:
    pixel_integration: true
    kernel: area
  noise:
    shot: true
    read_sigma: 0.01
    background: 0.02
  artifacts:
    cosmic_ray_probability: 0.0
    saturation: false

models:
  baselines: [bicubic, lanczos, wiener]
  pretrained:
    - name: documented_pretrained_model
      checkpoint: path/or/registry/reference
      scale: 2

ensemble:
  enabled: true
  samples: 32
  stochasticity:
    source: controlled_test_time_variation
    seed_stride: 1000
  retain_members: true
  filtering:
    method: mad
    threshold: 3.5
  consensus:
    method: median
  uncertainty:
    maps: [std, mad, percentile_interval]

recursion:
  enabled: true
  factors: [2, 2]
  max_depth: 2
  gates:
    min_agreement: 0.85
    max_relative_uncertainty: 0.20
    require_forward_consistency: true
    require_photometric_check: true
    max_metric_regression: 0.02

evaluation:
  metrics: [psnr, ssim, flux_error, centroid_error, morphology_error]
  bootstrap_replicates: 1000
  save_error_maps: true
  compare_to_reference: true

logging:
  save_config: true
  save_environment: true
  export_elabftw: true
  include_failure_cases: true
```

Values above are examples, not scientific constants. The agent must not silently treat them as validated thresholds. Thresholds should be tuned on a validation split and frozen before final test evaluation.

---

## 18. CLI and pipeline stages

Provide a single discoverable CLI with subcommands similar to:

```text
astrsr validate-config --config configs/p0_smoke.yaml
astrsr prepare-data --config ...
astrsr make-degradations --config ...
astrsr run-baselines --config ...
astrsr sample-ensemble --config ...
astrsr build-consensus --config ...
astrsr evaluate --config ...
astrsr run-recursion --config ...
astrsr run-ablation --config ...
astrsr report --run-id ...
astrsr export-elabftw --run-id ...
```

Requirements:

- `--config` is required for canonical experiments;
- `--seed`, `--device`, and `--dry-run` may override safe runtime settings but the override must be recorded;
- commands should fail loudly on missing checkpoints, unknown parameters, invalid shapes, NaN/Inf, or missing provenance;
- a smoke-test command must finish quickly on fixtures and exercise the full orchestration path;
- a resume mode may reuse immutable intermediates only when their configuration/data hashes match;
- output paths must never overwrite a completed run without explicit, logged permission.

The canonical experiment should be runnable through one documented command, while subcommands support debugging and ablations.

---

## 19. Testing strategy

### Unit tests

Test configuration validation, seed handling, PSF and noise operators, shape and dtype preservation, normalization, manifest checksums, alignment, robust statistics, metric edge cases, gate logic, and serialization.

### Integration tests

Run a tiny synthetic sample through degradation, baseline inference, ensemble generation, consensus, recursive gating, evaluation, and eLabFTW export. Verify that expected artifacts exist and that rerunning with the same seed produces the same result within documented tolerance.

### Scientific sanity tests

- forward degradation of an identity or simple point source behaves as expected;
- a no-noise/no-blur synthetic case does not reward invented detail;
- a deliberately unstable ensemble produces high disagreement and fails the gate;
- a deliberately outlier-contaminated ensemble records filtered members;
- a flux-preserving operator remains approximately flux-preserving under its documented conditions;
- a known centroid shift is detected;
- a recursion limit stops at the configured depth;
- metrics fail safely for incompatible units, shapes, masks, or dynamic ranges.

Do not treat passing software tests as evidence that the scientific hypotheses are true. They only establish implementation correctness for the tested behavior.

---

## 20. Experiment workflow

1. Write or update the hypothesis, primary metrics, dataset split, and planned thresholds.
2. Create a versioned configuration and run ID.
3. Validate the configuration and data manifest.
4. Run the P0 smoke test.
5. Run baselines on the validation and test partitions with identical inputs.
6. Run the ensemble sample-count pilot.
7. Freeze the selected sample count and gate thresholds based on validation results.
8. Run the untouched test set once for the primary comparison.
9. Run predeclared ablations and failure-case analyses.
10. Generate the report and eLabFTW export.
11. Review whether the evidence supports, weakens, or rejects each hypothesis.
12. Decide whether to expand data, seek expert review, or train a custom model.

For every major result, save both a compact summary and the underlying machine-readable data. A result that cannot be traced from report to configuration to artifact is not publication-ready.

---

## 21. Milestone plan

### Milestone 0 — Project scaffold and scientific contract

Create the repository structure, configuration validation, run records, seed handling, and this document. Define the first data manifest and P0 smoke fixture.

### Milestone 1 — Forward model and baseline evidence

Implement synthetic degradation with PSF, sampling, and noise. Add bicubic/Lanczos and one classical method. Confirm that the evaluation framework can recover known changes and report realistic failure modes.

### Milestone 2 — Existing pretrained SR inference

Integrate at least one existing pretrained SR model with documented checkpoint/license/device handling. Compare it fairly against interpolation and classical restoration.

### Milestone 3 — Stochastic ensemble and consensus

Implement controlled stochastic variation, configurable sample counts, output alignment, filtering, consensus, and uncertainty/disagreement maps. Run the sample-efficiency study.

### Milestone 4 — Recursive gate

Implement conservative one-step and multi-step recursion, explicit gate decisions, forward consistency checks, and recursion-off/unconditional/gated ablations.

### Milestone 5 — Validation package

Run held-out ground-truth evaluation, astronomy-relevant metrics, calibration/selective-prediction analyses, failure cases, and reproducibility checks. Target rapid completion of the pre-training algorithm and first evidence package in approximately one week with AI coding assistance, subject to actual data and hardware availability.

### Milestone 6 — Expert review decision

Prepare a concise methods/results/limitations package for review and mentorship with Dr. Kevin Crosby. Ask specifically for feedback on instrument realism, scientific metrics, domain assumptions, and whether the observed signal merits further work.

### Milestone 7 — Optional custom training

Only if justified, assemble paired/degraded training data, document the GPU/compute budget, train an astronomy-specific model, and repeat the complete evaluation and ablations.

### Milestone 8 — Paper and dissemination

Write a paper that reports the method, baselines, metrics, uncertainty behavior, failure cases, negative results, limitations, and reproducibility artifacts. Decide on public release only after reviewing data/model licenses and IP considerations.

---

## 22. IP, patent, and publication note

The project has publication intent, but novelty and patentability are unknown and must not be assumed. Before public disclosure of implementation details, code, datasets, or results that may have commercial value:

- discuss the disclosure plan with the relevant institution, employer, technology-transfer office, or legal counsel;
- perform an appropriate prior-art and novelty review;
- document inventorship and contribution where applicable;
- verify all dataset, checkpoint, and software licenses;
- decide what can be released, when, and under which license.

This document is a research/engineering specification, not legal advice. Do not promise patentability or delay ordinary reproducibility work without a specific reason and responsible review.

---

## 23. Definition of done

The project has reached a meaningful initial completion when:

- the P0 pipeline runs from configuration to report on a clean, documented fixture;
- at least the required baselines and the proposed ensemble/consensus method are compared fairly;
- stochasticity is real, controlled, and reproducible;
- consensus and uncertainty outputs are saved and evaluated;
- recursion can pass, fail, and stop for explicit reasons;
- held-out ground truth is used wherever available;
- photometric, astrometric, and morphological fidelity are measured alongside PSNR/SSIM;
- ablations test whether any benefit comes from ensemble sampling, consensus, uncertainty gating, or recursion;
- failure cases and limitations are visible in the report;
- data provenance, seeds, configs, checkpoints, software versions, and artifacts are recorded;
- an eLabFTW-ready export exists;
- the conclusion is framed as a measured reconstruction/inference result under stated conditions, not as recovery of arbitrary unseen detail;
- a documented decision is made about expert review, further data, custom training, or stopping.

The project is allowed to conclude that the method does not provide a reliable scientific advantage. A well-supported negative result satisfies the research objective better than an attractive but unvalidated image.

---

## 24. Instructions to the autonomous coding agent

When implementing this project:

1. Read this document before making architectural or scientific changes.
2. Preserve raw/reference inputs and never edit files under a read-only source directory.
3. Prefer small, testable modules and explicit configuration over hidden defaults.
4. Implement P0 end to end before adding custom training, web UI, optimization, or large-scale infrastructure.
5. Record every stochastic source, threshold, model checkpoint, and data transformation.
6. Keep scientific outputs separate from display-only images.
7. Make the pipeline fail loudly when provenance, units, shapes, or checkpoints are missing.
8. Add or update tests for every nontrivial behavior, especially gate logic and forward-model assumptions.
9. Do not use visual plausibility as the acceptance criterion for a scientific improvement.
10. If a requirement is ambiguous, choose the smallest reversible implementation and record the assumption in the run/report; do not silently expand scope.
11. If a result appears unusually strong, attempt to disprove it with leakage checks, degradation-mismatch tests, independent seeds, and expert review.
12. State clearly when a result is synthetic-only, model-dependent, uncalibrated, or not supported by ground truth.

The guiding principle is: **the system may make a measured, testable inference more useful, but it must never present model imagination as newly observed astronomical information.**
