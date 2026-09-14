# Real Chart Assets

Each sample is self-contained under `{source_family}/{sample_id}/`:

```
assets/real_charts/
├── fred/                     # Financial time-series (grid_aware)
│   ├── fred_001/
│   │   ├── meta.json        # Sample metadata, question, curated facts
│   │   └── image.png        # Chart screenshot
│   └── ...
├── engisphere/               # Engineering charts (grid_aware)
│   ├── engisphere_001/
│   │   ├── meta.json
│   │   ├── image.png
│   │   └── state.json       # Generator state (engisphere only)
│   └── ...
└── omniscience/              # Scientific charts (grid_free)
    └── omniscience_001/
        ├── meta.json         # m0_mode: grid_free (LLM extracts evidence)
        └── image.png         # No reference.csv needed for grid-free samples
```

## M0 Modes

Each sample's `meta.json` specifies an `m0_mode` field:

- **`grid_aware`** (default): Pre-curated `candidate_facts`, `verified_facts`,
  `useful_facts` in meta.json simulate CV-based coordinate extraction (M0).
  Used for charts with grid lines (FRED, EngiSphere).

- **`grid_free`**: LLM vision extracts visual evidence directly from the image
  at runtime. No pre-curated facts needed. Used for general scientific charts
  without grid lines (OmniScience).

## Adding a new sample

1. Create `{source_family}/{sample_id}/` directory
2. Place `image.png` and `meta.json` (required)
3. Set `m0_mode` in meta.json (`grid_aware` or `grid_free`)
4. The presenter auto-discovers all samples via directory scan

## Capture rules (grid-aware samples)

- Open the source page and set the target date/value window
- Crop to the chart region only
- Keep axes, tick labels, and grid lines visible
- Close tooltips before taking the screenshot
- Store curated visible-window facts in `meta.json`

## Ground Truth Format

The current committed asset set does not include per-sample `reference.csv`
files. Ground truth for these samples is stored in `meta.json`, including the
question, gold answer, answer validation rule, and any curated facts used by the
presenter path. These assets are therefore suitable for task-level presenter
diagnostics, not for point-level extrema validation unless additional
coordinate-level annotations are created in a separate, provenance-tracked pass.

## Data Sources and Attribution

### FRED (Federal Reserve Economic Data)

Charts sourced from the [Federal Reserve Economic Data](https://fred.stlouisfed.org) portal, a public-domain repository maintained by the Federal Reserve Bank of St. Louis. FRED data is in the public domain and free to use.

### EngiSphere

Stress-strain curves generated using the [EngiSphere Stress-Strain Curve Generator](https://engisphere.com/calculators/stress-strain-curve-generator/), covering diverse materials under varying temperature conditions.

### OmniScience (Nature Communications, CC-BY 4.0)

Scientific figures drawn from open-access publications in *Nature Communications*, licensed under [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Per the license terms, original authors and DOIs are attributed below.

| Sample | Title | DOI |
|--------|-------|-----|
| omniscience_001 | Engineering hybrid nanotube wires for high-power biofuel cells | [10.1038/ncomms1000](https://doi.org/10.1038/ncomms1000) |
| omniscience_002 | Structural diversity in binary superlattices self-assembled from polymer-grafted nanocrystals | [10.1038/ncomms10052](https://doi.org/10.1038/ncomms10052) |
| omniscience_003 | Tracing the fate of carbon and the atmospheric evolution of Mars | [10.1038/ncomms10003](https://doi.org/10.1038/ncomms10003) |
| omniscience_004 | The effect of soil-borne pathogens depends on the abundance of host tree species | [10.1038/ncomms10017](https://doi.org/10.1038/ncomms10017) |
| omniscience_005 | Experimental investigation of classical and quantum correlations under decoherence | [10.1038/ncomms1005](https://doi.org/10.1038/ncomms1005) |
| omniscience_006 | A small-gap electrostatic micro-actuator for large deflections | [10.1038/ncomms10078](https://doi.org/10.1038/ncomms10078) |
| omniscience_007 | Jurassic climate mode governed by ocean gateway | [10.1038/ncomms10015](https://doi.org/10.1038/ncomms10015) |
| omniscience_008 | Atlantic multi-decadal oscillation covaries with Agulhas leakage | [10.1038/ncomms10082](https://doi.org/10.1038/ncomms10082) |
| omniscience_009 | High-efficiency polymer solar cells with small photon energy loss | [10.1038/ncomms10085](https://doi.org/10.1038/ncomms10085) |
| omniscience_010 | Large microwave generation from current-driven magnetic vortex oscillators in magnetic tunnel junctions | [10.1038/ncomms1006](https://doi.org/10.1038/ncomms1006) |
