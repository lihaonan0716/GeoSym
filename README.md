# GeoSym

**GeoSym: Explicit Evidence Construction and Layered Reasoning for Quantitative Chart Understanding**

Answering a quantitative question about a chart requires recovering values encoded in pixel geometry and then computing with them. An end-to-end multimodal model does both in a single model call, so a wrong answer does not reveal which operation failed.

GeoSym separates the two. An **evidence extraction stage (EX)** recovers measured quantities from the image by geometric analysis, and a **layered reasoning stage (LR)** consumes them through reading, analysis and inference layers, each of which records its output before the next layer runs. This repository contains the pipeline, the baselines, the evaluation code and the scripts that produced the reported results.

**Keywords:** chart understanding · multimodal reasoning · quantitative visual reasoning · vision-language models · evidence extraction

## Evidence extraction — examples

On GeoSym-Bench charts, which carry a regular integer grid, the extractor detects the grid lattice and the curve, and keeps only curve points that lie on lattice intersections. Original benchmark images (left) and the extracted coordinates (right):

<table>
<tr>
<th>Function Family</th>
<th>Original Image</th>
<th>Extracted coordinates</th>
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

## Architecture

### EX: evidence extraction

EX is defined by an evidence interface (`evaluator/evidence/`) rather than by a fixed algorithm. Two providers are evaluated, one per chart population:

- **Grid-geometric extractor** (`recognizer/core/clean_grid_fallback.py`, entry point `extract_clean_cartesian_grid`), used on GeoSym-Bench. It reads no text. Four steps: lattice detection from a grey-pixel mask and column/row projections; axis location from near-black projections; curve isolation by HSV thresholding inside the lattice region; integer snapping with refusal, where a candidate further than a quarter grid unit from an integer is discarded rather than rounded. Extraction fails closed when either axis is missing or fewer than three grid lines are found in either direction.
- **PlotQA chain** (`evaluator/reasoning/plotqa_m0_chain.py`), used on real statistical charts without coordinate grids: category-axis resolution, a least-squares value-axis refit over frozen PaddleOCR tick boxes, and mark admission.

The GeoSym-Bench runners select the grid-geometric extractor through the `clean_grid_only_m0()` context manager; the broader OCR-based route under `recognizer/` is not used by the reported GeoSym-Bench results. Admitted facts are injected as they are: there is no reliability scoring, tiering, gating or re-ranking.

### LR: layered reasoning

Following Curcio's graph-comprehension taxonomy, LR runs three layers:

- **L1 (reading)** receives the image and the extracted evidence, **never the task parameters**, and records coordinates, intercepts and extrema. A reader that knows the target finds it in the pixels, so withholding the task keeps the reading independent; this is enforced by construction, with no parameter to pass it through.
- **L2 (analysis)** receives the image, the L1 output and the task parameters (prediction x-positions, rate-of-change interval), and computes predictions, rate of change, monotonicity and symmetry.
- **L3 (inference)** receives the image and the L1 and L2 outputs, and infers the closed-form expression, domain and range.

Layering costs three reasoning calls per sample against one for a single pass. Its contribution is an inspectable record of what was read, computed and inferred; it is not claimed to improve accuracy (see Results).

## GeoSym-Bench

A controlled benchmark of synthetic function charts with exact algebraic ground truth.

- **Data:** [huggingface.co/datasets/lihaonan0716/GeoSym-Bench](https://huggingface.co/datasets/lihaonan0716/GeoSym-Bench) (CC-BY 4.0). The released files are the canonical benchmark: a generated pool of 480 charts, 60 per family, 1440×1440 PNG with one ground-truth JSON each.
- **Evaluated roster:** [`benchmark/roster_240.json`](benchmark/roster_240.json) lists the 240 samples behind the reported GeoSym-Bench results, with per-family counts (19–32) and SHA-256 checksums for every image and ground-truth file.

### Function families

| Category | Family | General Form |
|----------|--------|--------------|
| Elementary | `linear` | y = ax + b |
| Elementary | `quadratic` | y = a(x−h)² + k |
| Elementary | `absolute` | y = a\|x−h\| + k |
| Polynomial | `cubic` | y = a(x−h)³ + b(x−h) + k |
| Piecewise | `piecewise_linear` | two affine branches joined at (h, k) |
| Transcendental | `exponential` | y = a·b^(x−h) + k |
| Transcendental | `logarithmic` | y = a·log₂(x−h) + k |
| Special | `rational` | y = a/(x−h) + k |

### Evaluation dimensions

Ten dimensions, combined by the three-group weighting in `config/default_config.yaml` (`reasoning.layers`). They are **not** equally weighted: effective per-dimension weights range from 0.045 to 0.150. D1–D9 are scored against exact ground truth; D10 is scored by a judge model that differs from the reasoning model. Extracted coordinates are intermediate data and are not scored.

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
| D10: Inference quality | G3 mathematical inference (0.40) | 0.120 | Scored by the judge model |

## Experimental conditions

All GeoSym-Bench conditions use the same reasoning backbone (GLM-4.6V) and judge (Qwen3-VL-Plus), at temperature 0.

| ID | Condition | CoT | Prompt variants | OCR evidence | GeoSym | Reasoning calls / sample |
|----|-----------|:---:|:---------------:|:------------:|:------:|:------------------------:|
| E0 | DirectQA | | | | | 1 |
| E1 | DirectQA-CoT | ✓ | | | | 1 |
| E2 | DirectQA-SC | ✓ | ✓ | | | 3 |
| E3 | OCRRulesQA | | | ✓ | | 1 |
| E4 | OCRRules-CoT | ✓ | | ✓ | | 1 |
| **`E5_raw`** | **GeoSym** (EX + LR) | | | | ✓ | 3 |

- E2 aggregates over paraphrased prompt variants by majority vote at `T=0`. It is deterministic, unlike stochastic self-consistency sampling.
- E3/E4 inject evidence read from tick text by local PaddleOCR (3.7.0, `PP-OCRv6_medium`) with rule-based feature computation. GeoSym against E3/E4 therefore contrasts two ways of acquiring a scale: chart geometry against recognised text.
- The condition string is `E5_raw` in code, configs and result files. Historical result files also contain a retired arm named `E5`; filter on `E5_raw`, never on `E5`.

## Results

**GeoSym-Bench** (GLM-4.6V, judge Qwen3-VL-Plus). Condition means over each condition's scored samples:

| Condition | Mean score | Scored n |
|-----------|:----------:|:--------:|
| E0 DirectQA | 68.65 | 240 |
| E1 DirectQA-CoT | 69.88 | 239 |
| E2 DirectQA-SC | 70.64 | 240 |
| E3 OCRRulesQA | 59.75 | 229 |
| E4 OCRRules-CoT | 57.11 | 229 |
| **GeoSym (`E5_raw`)** | **81.46** | 237 |

- **Headline contrast:** on the 237 samples scored under both GeoSym and E2, the strongest baseline at the same three reasoning calls per sample, GeoSym leads by a paired mean of **+10.66** points (family-clustered bootstrap 95% CI [+4.90, +16.76]).
- **All-objective-zero rows:** none among GeoSym's 237 scored samples, against 6.2–11.8% across the scored baseline samples.
- **The OCR baselines score below direct prompting.** Adding extracted evidence alone does not ensure a gain.
- **Layering does not improve accuracy.** With extracted evidence held byte-identical, layered and single-pass reasoning obtain similar mean scores: paired difference +0.12 (95% CI [−2.78, +3.02], p = 0.935, n = 104). The observed gain is associated primarily with evidence construction; layering supplies separately inspectable intermediate records at three reasoning calls against one.

**PlotQA** (real statistical charts without coordinate grids):

- **Evidence coverage:** across 499 charts, mean numerical-value coverage rises from 0.522 to 0.765 under the repaired extraction chain. The gain is concentrated in horizontal bar charts (0.035 → 0.779); line and vertical-bar charts drift slightly down.
- **Answer accuracy:** on 80 paired samples from two numeric strata (GPT-5.5), measured evidence plus the image shows no significant difference from image-only prompting: 0.588 against 0.637, McNemar p = 0.388. Higher coverage does not by itself improve downstream question answering.

## Reproducing the experiments

Paid runners default to a dry run that builds and checks the frozen plan without network calls; spending requires an explicit flag.

| Result | Script |
|--------|--------|
| GeoSym-Bench condition sweep (E0–E4, `E5_raw`) | `scripts/run_glm46v_geosym_bench_240x7.py` (`--roster-file`, `--run-paid`) |
| Identical-evidence layering comparison | `scripts/run_m3_attribution_pilot.py`, rescored by `scripts/rescore_m3_attribution_json_repair.py` |
| PlotQA extraction coverage | `scripts/verify_plotqa_m0_chain_wiring.py` (zero cost) |
| PlotQA answer accuracy | `scripts/run_plotqa_numeric_attribution.py` (`--execute`), analysed by `scripts/analyze_plotqa_numeric_attribution.py` |
| Extraction-interface applicability | `scripts/evaluate_evidence_provider_substitutability.py` (zero cost) |

The GeoSym-Bench runner reads the dataset from `dataset/images/` and `dataset/ground_truth/`; download it from Hugging Face into `dataset/`. PlotQA scripts expect the official PlotQA images and annotations, which are not redistributed here.

## Repository structure

| Directory | Role |
|-----------|------|
| `recognizer/` | Grid-geometric extractor (`core/clean_grid_fallback.py`) and the broader OCR-based extraction route |
| `evaluator/` | Evidence interface, PlotQA extraction chain, model adapters, scoring, baselines, response parsing |
| `app/` | Layered-reasoning orchestration, caching and response schemas |
| `prompts/` | Prompt templates for every condition, the judge and the PlotQA arms |
| `generator/` | Synthetic function-chart generator |
| `presenter/` | Tick-text ruler and LLM-located extraction for real-world charts, with demonstration assets |
| `benchmark/` | Evaluated GeoSym-Bench roster with checksums |
| `scripts/` | Experiment runners, rescoring and analysis |
| `tests/` | Test suite |

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy `config/api_keys.yaml.example` to `config/api_keys.yaml` and add keys for the providers you use. GPT and Claude adapters read an API-compatible gateway base URL from `ninerouter_base_url`, `NINEROUTER_BASE_URL`, `GEOSYM_OPENAI_BASE_URL` or `GEOSYM_CLAUDE_BASE_URL`. Model identifiers live in `config/default_config.yaml`.

The E3/E4 baselines need local PaddleOCR: install the `paddlepaddle` wheel for your device, then `pip install paddleocr>=3.7.0`.

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests that need the GeoSym-Bench files skip when `dataset/` is absent.

## Generating new charts

`generator/` produces synthetic function charts of the same kind as GeoSym-Bench:

```bash
python scripts/run_generate.py --count 20 --seed 42 --out-dir my_charts
```

The released benchmark files are the reference for the reported results. The generator in this repository does not reproduce them byte for byte, so use the Hugging Face files, checked against `benchmark/roster_240.json`, to reproduce the paper.

## Supported model providers

| Provider | Models | Adapter |
|----------|--------|---------|
| ZhipuAI | GLM-4.6V | `evaluator/adapters/glm.py` |
| Alibaba | Qwen3-VL-Plus, Qwen3.5-Plus | `evaluator/adapters/qwen.py` |
| OpenAI-compatible gateway | GPT-5.5 | `evaluator/adapters/openai.py` |
| Google | Gemini | `evaluator/adapters/gemini.py` |
| Anthropic-compatible gateway | Claude | `evaluator/adapters/claude.py` |
| Moonshot | Kimi | `evaluator/adapters/kimi.py` |

## License

The code is released under the [MIT License](LICENSE). GeoSym-Bench is released separately on [Hugging Face](https://huggingface.co/datasets/lihaonan0716/GeoSym-Bench) under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). The third-party figures in `assets/real_charts/` are not covered by the MIT License; each is distributed under the licence of its source article, listed in [`assets/real_charts/README.md`](assets/real_charts/README.md).

## Citation

```bibtex
@misc{wang2026geosym,
  title  = {GeoSym: Explicit Evidence Construction and Layered Reasoning for Quantitative Chart Understanding},
  author = {Wang, Yongqing and Zhang, Qisheng and Guo, Linyan and Li, Haonan and Sun, Tianjun and Yu, Samson S.},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/lihaonan0716/GeoSym}
}
```
