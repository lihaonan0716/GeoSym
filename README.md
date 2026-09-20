# GeoSym

**GeoSym: Geometric Evidence Extraction and Layered Reasoning for Quantitative Chart Understanding**

Multimodal large language models have demonstrated remarkable capabilities in visual understanding, yet they remain prone to hallucination and imprecise numerical reasoning when confronted with tasks demanding quantitative accuracy. Existing enhancement strategies — including chain-of-thought prompting, self-consistency decoding, and tool-augmented reasoning — improve general reasoning but leave evidence construction implicit in the prompt, so the quality of what the reasoner is given is neither engineered nor auditable.

GeoSym is a chart-evidence pipeline for quantitative reasoning that makes evidence construction an explicit pipeline stage feeding a layered reasoning chain. The framework operates through two modules:

1. **M0 — Geometric Evidence Extraction**: Computation-grounded visual extraction through geometric pixel-level analysis, producing coordinates, intercepts, extrema, and monotonicity intervals.
2. **M3 — Layered Reasoning**: Layered reasoning that presents structured evidence across reading, analysis, and inference layers aligned with Curcio's graph comprehension taxonomy, each layer receiving a layer-restricted evidence view.

> **Two modules, not four.** Earlier versions of this framework included **M1** (confidence-scored fact construction) and **M2** (utility-aware evidence selection). Both were **retired on 2026-09-19** after eight independent evaluations found no discriminative signal: held-out AUC fell to 0.584 as the underlying evidence improved, all eight substitute features stayed below 0.60, a seven-point budget sweep moved the score by at most +2, every per-fact suppression rule was net-negative, and the two arms produced byte-identical output on 76.8% of samples. The current pipeline performs **no reliability scoring, tiering, gating, or re-ranking** — every admitted fact is injected. Results produced under the retired modules are retained below and labelled as historical.

**Keywords:** Multimodal reasoning · Quantitative chart reasoning · Vision-language models · Visual evidence extraction · Layered reasoning

## M0 Visual Evidence Extraction — Examples

GeoSym's M0 module performs computation-grounded coordinate extraction via geometric pixel-level analysis. Below are examples from four function families showing original benchmark images (left) and M0 annotated outputs (right) with detected curve-grid intersection points:

<table>
<tr>
<th>Function Family</th>
<th>Original Image</th>
<th>M0 Annotated Output</th>
</tr>
<tr>
<td><b>Linear</b><br><code>y = ax + b</code></td>
<td><img src="assets/readme/linear_original.png" width="360"></td>
<td><img src="assets/readme/linear_annotated.png" width="360"></td>
</tr>
<tr>
<td><b>Quadratic</b><br><code>y = a(x-h)² + k</code></td>
<td><img src="assets/readme/quadratic_original.png" width="360"></td>
<td><img src="assets/readme/quadratic_annotated.png" width="360"></td>
</tr>
<tr>
<td><b>Exponential</b><br><code>y = a·b^(x-h) + k</code></td>
<td><img src="assets/readme/exponential_original.png" width="360"></td>
<td><img src="assets/readme/exponential_annotated.png" width="360"></td>
</tr>
<tr>
<td><b>Cubic</b><br><code>y = a(x-h)³ + b(x-h) + k</code></td>
<td><img src="assets/readme/cubic_original.png" width="360"></td>
<td><img src="assets/readme/cubic_annotated.png" width="360"></td>
</tr>
</table>

Each annotated image shows detected intersection points (green/yellow circles) with their computed real-valued coordinates. These coordinate anchors are the evidence set that M3 consumes.

## Framework Architecture

GeoSym enforces a clear separation of concerns: a computation-grounded module (M0) produces evidence, and the reasoning module (M3) synthesizes across abstraction layers. Each module operates within its domain of competence, and each is defined by an evidence interface rather than a fixed algorithm, so implementations can be substituted.

### M0: Candidate Evidence Generation

M0 produces a candidate evidence set through deterministic, pixel-grounded visual extraction. The design principle is **competence separation**: OCR provides axis scale information, geometric computation establishes pixel-to-coordinate mappings, and visual analysis identifies data point locations. The framework is agnostic to the specific extraction algorithm — M0 defines an interface, allowing adaptation to different visual domains.

Two complementary implementations share a common ruler infrastructure:

- **Function-Specialized M0** (`recognizer/`): A deterministic geometric pipeline for charts with integer coordinate grids. Detects grid lines via Hough transform, establishes pixel-to-coordinate mapping, and identifies curve-grid intersection points. Stage 2.5 adds least-squares regression cross-validation using the shared ruler module.
- **Enhanced M0** (`presenter/`): A four-phase pipeline for real-world charts without integer grids. Phase 1 builds a coordinate ruler from OCR tick labels via least-squares regression. Phase 2 uses an LLM to locate semantically important points at the pixel level. Phase 3 converts pixel coordinates to chart units through the ruler. Phase 4 checks measurements against axis bounds and fuses evidence into a standardized format.
- **Shared Ruler** (`presenter/ruler.py`): OCR-based coordinate system construction used by both implementations. Supports linear, logarithmic, datetime, and categorical axis scales.

### Fact Construction

From the extracted coordinates, GeoSym infers intercepts, extrema, monotonicity intervals, and symmetry, then formats them as a single flat JSON block for injection. Facts are emitted **unannotated**: there are no confidence tiers, no scores, and no selection step. Flat injection is the only surviving profile — the tiered profile was removed with M1/M2.

### M3: Layered Reasoning

M3 consumes the evidence through a three-layer reasoning hierarchy:

- **L1 (Reading)**: Extract or consolidate coordinates, intercepts, and extrema. Evidence is injected primarily at this layer.
- **L2 (Analysis)**: Perform quantitative analysis — function value predictions, rate of change, monotonicity, and symmetry.
- **L3 (Inference)**: Derive higher-order mathematical conclusions — closed-form expression, domain, and range.

Evidence injection at L1 ensures that stronger visual grounding propagates upward into L2 and L3, where abstract reasoning benefits from explicit, auditable anchors rather than implicit visual impressions alone.

## GeoSym-Bench

GeoSym-Bench is a controlled evaluation benchmark with complete ground-truth annotations across **8 function families**. The generated cohort contains **480 samples**; historical experiments are reported on a **460-sample complete-case subset** where all compared conditions completed successfully.

The dataset is hosted on HuggingFace: [lihaonan0716/GeoSym-Bench](https://huggingface.co/datasets/lihaonan0716/GeoSym-Bench)

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="lihaonan0716/GeoSym-Bench", repo_type="dataset")
```

### Function Families

| Category | Family | General Form |
|----------|--------|--------------|
| Elementary | `linear` | y = ax + b |
| Elementary | `quadratic` | y = a(x−h)² + k |
| Elementary | `absolute` | y = a\|x−h\| + k |
| Polynomial | `cubic` | y = a(x−h)³ + b(x−h) + k |
| Polynomial | `piecewise_linear` | y = sᵢ(x−hᵢ) + kᵢ |
| Transcendental | `exponential` | y = a·b^(x−h) + k |
| Transcendental | `logarithmic` | y = a·log₂(x−h) + k |
| Special | `rational` | y = a/(x−h) + k |

### Evaluation Dimensions

Each image carries annotations across 10 evaluation dimensions. The dimensions are **not** equally weighted: they are combined by a three-group scheme defined in `config/default_config.yaml`, giving effective per-dimension weights from 0.045 to 0.150.

| Dimension | Group (group weight) | Effective weight | Description |
|-----------|----------------------|:----------------:|-------------|
| D1: Intercepts | G1 visual readout (0.30) | 0.150 | x- and y-intercepts |
| D2: Extrema | G1 visual readout (0.30) | 0.150 | Local minima/maxima with type labels |
| D3: Predictions | G2 quantitative analysis (0.30) | 0.075 | Function values at specified non-integer x-values |
| D4: Rate of change | G2 quantitative analysis (0.30) | 0.045 | Average slope over a defined interval |
| D5: Monotonicity | G2 quantitative analysis (0.30) | 0.105 | Increasing/decreasing intervals |
| D6: Symmetry | G2 quantitative analysis (0.30) | 0.075 | Even, odd, or neither |
| D7: Expression | G3 mathematical inference (0.40) | 0.120 | Closed-form formula |
| D8: Domain | G3 mathematical inference (0.40) | 0.080 | Mathematical domain |
| D9: Range | G3 mathematical inference (0.40) | 0.080 | Mathematical range |
| D10: Inference quality | G3 mathematical inference (0.40) | 0.120 | Independently scored by judge model |

Extracted coordinates are M0 intermediate data and are deliberately not scored.

## Experimental Conditions

Six live conditions form a progressive enhancement spectrum:

| ID | Condition | Image | CoT | SC | OCR | GeoSym | Description |
|----|-----------|:-----:|:---:|:--:|:---:|:------:|-------------|
| E0 | DirectQA | ✓ | | | | | Raw image → MLLM single-pass |
| E1 | DirectQA-CoT | ✓ | ✓ | | | | With chain-of-thought decomposition |
| E2 | DirectQA-SC | ✓ | ✓ | ✓ | | | Aggregation over prompt variants, deterministic at `T=0` |
| E3 | OCRRulesQA | ✓ | | | ✓ | | Mathpix OCR extraction + rule-based feature computation |
| E4 | OCRRules-CoT | ✓ | ✓ | | ✓ | | Mathpix OCR + rule-based features + CoT |
| **`E5_raw`** | **GeoSym** | ✓ | | | | ✓ | **M0 + M3 — the method itself, not an ablation** |

The retired `E5` (FullGeoSym, M0+M1+M2+M3) is no longer run. Its rows remain in the historical artifacts alongside `E5_raw` rows, so the condition identifier `E5_raw` is kept byte-identical in code, configs, and data even though the prose name is simply "GeoSym".

E2 aggregates over prompt variants at `T=0` rather than sampling stochastically; it is named "self-consistency" for continuity with the literature, but it is deterministic.

## Cross-Domain Evaluation

Beyond the controlled benchmark, GeoSym includes real-world chart demonstrations across three domains:

| Dataset | Source | License | Domain |
|---------|--------|---------|--------|
| FRED | [Federal Reserve Economic Data](https://fred.stlouisfed.org) | Public domain | Financial time-series |
| EngiSphere | [EngiSphere Stress-Strain Generator](https://engisphere.com/calculators/stress-strain-curve-generator/) | — | Engineering stress-strain diagrams |
| OmniScience | [Nature Communications](https://www.nature.com) | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Scientific figures |

OmniScience figures are reproduced under Creative Commons Attribution 4.0 (CC-BY 4.0). Original paper DOIs and full attribution are provided in [`assets/real_charts/README.md`](assets/real_charts/README.md).

M3 operates without modification across domains; only M0 requires domain-specific instantiation via the Enhanced M0 pipeline (`presenter/`). These cases illustrate measurement traceability, feature localization, and source attribution rather than constituting a systematic external benchmark.

## Key Results

Across 460 evaluated samples with GPT-5.4 as the backbone MLLM:

| Condition | Mean Score | Δ vs. strongest baseline condition (E0) |
|-----------|:----------:|:----------------------------------------:|
| E0 (DirectQA) | 79.46 | — |
| E1 (DirectQA-CoT) | 78.88 | -0.58 |
| E2 (DirectQA-SC) | 78.81 | -0.65 |
| E3 (OCRRulesQA) | 78.17 | -1.30 |
| E4 (OCRRules-CoT) | 79.25 | -0.22 |
| **`E5_raw` (GeoSym, M0+M3)** | **93.48** | **+14.02** |
| ~~E5 (FullGeoSym, M0+M1+M2+M3)~~ | 93.67 | +14.21 |

- GeoSym (`E5_raw`) outperforms the strongest baseline condition (E0) by 14.02 points.
- The E5 row is **historical only** and is struck through because the modules that distinguished it from `E5_raw` have been retired. It is shown for continuity with earlier reports, not as a current result.
- The gap between the two arms is **+0.19 in E5's favour, not a significant main effect** — this is one of the eight findings that led to retiring M1/M2. The larger +2.42 effect on the 18% of samples where extraction quality degrades was selected by an outcome variable, so it was only ever reportable as a post-hoc diagnostic, never as confirmatory robustness evidence; it did not survive the fuller evaluation.
- Sample-wise figures reported previously (best-of-baselines margin of 10.67 points; 86% of samples improving over the per-sample baseline maximum; 93.3% improving over E0) were computed for the E5 arm and are likewise historical. Recomputing them for `E5_raw` requires a re-analysis that has not yet been run.

`E5_raw` merges M0 and M3 into a single arm, so these numbers do **not** separate the contribution of geometric extraction from that of layered reasoning. Disentangling the two is an open problem, not a settled result.

## Repository Structure

```text
GeoSym/
├── generator/       # Benchmark generation and mathematical ground truth
├── recognizer/      # Function-specialized M0 (grid-based coordinate extraction)
├── presenter/       # Enhanced M0 (ruler + LLM locate) + real-world chart showcase
├── evaluator/       # LLM adapters, metrics, fact construction, M3 reasoning
├── prompts/         # Prompt templates for all pipeline stages
├── app/             # Backend orchestration modules for reasoning, cache, and schemas
├── config/          # Configuration files and API key template
├── scripts/         # CLI tools
├── tests/           # Test suite
├── docs/            # Technical documentation
└── requirements.txt
```

| Directory | Role |
|-----------|------|
| `generator/` | Benchmark generation: synthetic function-graph images, perturbations, and exact ground truth |
| `recognizer/` | GeoSym's visual extraction backbone for coordinate evidence on controlled benchmarks |
| `presenter/` | Enhanced M0 pipeline (ruler, measure, verify) and real-world chart showcase flows |
| `evaluator/` | Fact construction and prompt formatting, LLM adapters, reasoning scoring, judge scoring, baselines, and response parsing |
| `app/` | Backend orchestration for reasoning pipelines, cache records, schemas, and connection checks |
| `prompts/` | Prompt templates for reasoning layers, judge evaluation, M0 locate, visual fact inference |
| `scripts/` | CLI tools for dataset generation, benchmark execution, analysis, audits, and paper evidence packaging |
| `tests/` | Pytest suite covering the public surface and the evidence-channel contracts |

## Setup

```bash
python -m pip install -r requirements.txt
```

Core dependencies include OpenCV, NumPy, Pillow, PyYAML, and SymPy.

OCR backend for the E3/E4 baselines and for OCR-derived axis/tick extraction:

- **PaddleOCR** (current) runs locally and needs no credentials. Install the
  `paddlepaddle` wheel matching your device per the upstream instructions, then
  `pip install paddleocr>=3.7.0`. Runs used PaddleOCR 3.7.0 with the
  `PP-OCRv6_medium` detection and recognition models. Scripts that expose a
  `--device` flag accept `cpu` or `gpu:0`; PaddleOCR keeps a model resident in
  VRAM, so prefer `cpu` when another GPU job is in flight.
- **Mathpix** (earlier runs) is a hosted API; configure credentials in
  `config/api_keys.yaml`. Results produced under it are not pooled with
  PaddleOCR results, since detector output differs.
- **EasyOCR** is the automatic fallback in the backend ladder for environments
  where `paddlepaddle` cannot be installed. It is not a reported baseline.

Any reported OCR figure states which backend produced it.

Copy `config/api_keys.yaml.example` to `config/api_keys.yaml` and fill in your API keys. GPT and Claude adapters use an API-compatible gateway base URL from `ninerouter_base_url`, `NINEROUTER_BASE_URL`, `GEOSYM_OPENAI_BASE_URL`, or `GEOSYM_CLAUDE_BASE_URL`; the public repository does not hardcode a private gateway URL.

## Testing

Install development dependencies and run tests through the active Python interpreter:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Using `python -m pytest` avoids accidentally invoking a globally installed `pytest` from a different Python environment.

## Supported LLM Providers

| Provider | Models | Adapter |
|----------|--------|---------|
| OpenAI-compatible gateway | GPT-5.5 | `evaluator/adapters/openai.py` |
| Google | Gemini-3.1-Pro, Gemini-3-Pro | `evaluator/adapters/gemini.py` |
| Anthropic-compatible gateway | Claude-Opus-4.8, Claude-Opus-5 | `evaluator/adapters/claude.py` |
| Alibaba | Qwen3.5-Plus, Qwen3-VL-Plus | `evaluator/adapters/qwen.py` |
| Moonshot | Kimi-K2.5, Moonshot-V1 | `evaluator/adapters/kimi.py` |
| ZhipuAI | GLM-4.6V | `evaluator/adapters/glm.py` |

All adapters use `temperature=0.0` for maximum determinism, except `Kimi-K2.5` (`temperature=1.0` per API constraint). OpenAI additionally sets `seed=42`.

## Configuration

Primary configuration: `config/default_config.yaml`

Key sections:

- `page`, `chart`, `coords`, `ticks`, `curves`, `noise`: image generation and perturbation
- `reasoning.layers`: evaluation dimension definitions
- `reasoning.intervention`: evidence injection settings (flat profile only)
- `reasoning.judge`: judge LLM configuration
- `models`: LLM provider and model registry
- `output`: dataset locations

## Generate Benchmark Data

```bash
python scripts/run_generate.py --count 20 --seed 42 --out-dir dataset
```

With a custom config:

```bash
python scripts/run_generate.py --count 50 --config config/my_config.yaml --out-dir dataset
```

CLI flags:

- `--count`: number of samples per subset
- `--config`: YAML config path
- `--seed`: reproducible seed
- `--out-dir`: target directory for `images/` and `ground_truth/`

## Ground Truth Layout

Each generated image has a paired JSON file containing:

- `chart_bbox`, `page_size`, `layout`
- `x_range`, `y_range`, tick metadata
- `intersections`
- `functions` (parameters, expressions, type)
- `question.ground_truth` (all 10 evaluation dimensions)
- `extended_properties` (intercepts, extrema, monotonicity, symmetry)
- `benchmark_meta` (function family, noise profile, seed)

## Minimal Programmatic Use

```python
from generator.config_loader import load_config
from generator.core import generate_one

cfg = load_config()
img, gt = generate_one(cfg, output_image_path="out.png", output_gt_path="out.json", seed=1)
```

## License

This repository is licensed under the [MIT License](LICENSE). The GeoSym-Bench dataset is hosted separately on [HuggingFace](https://huggingface.co/datasets/lihaonan0716/GeoSym-Bench) under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use GeoSym in your research, please cite:

```bibtex
@misc{wang2026geosym,
  title     = {GeoSym: Geometric Evidence Extraction and Layered Reasoning for Quantitative Chart Understanding},
  author    = {Wang, Yongqing and Zhang, Qisheng and Guo, Linyan and Li, Haonan and Sun, Tianjun and Yu, Samson S.},
  year      = {2026},
  note      = {Manuscript in preparation. Software and dataset available online},
  url       = {https://github.com/lihaonan0716/GeoSym}
}
```

> The citation will be updated with the published DOI upon acceptance.
