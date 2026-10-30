# Real Chart Assets

Demonstration inputs for the `presenter/` path on real-world charts. These
samples are not part of GeoSym-Bench and are not used by any result reported
in the paper.

Each sample is self-contained under `{source_family}/{sample_id}/`:

```
assets/real_charts/
└── omniscience/              # Scientific charts (grid_free)
    └── omniscience_002/
        ├── meta.json         # m0_mode: grid_free (LLM extracts evidence)
        └── image.png
```

## M0 Modes

Each sample's `meta.json` specifies an `m0_mode` field:

- **`grid_aware`** (default): Pre-curated `candidate_facts`, `verified_facts`,
  `useful_facts` in meta.json simulate CV-based coordinate extraction (M0).
  Used for charts with grid lines.

- **`grid_free`**: LLM vision extracts visual evidence directly from the image
  at runtime. No pre-curated facts needed. Used for general scientific charts
  without grid lines.

## Adding a new sample

1. Create `{source_family}/{sample_id}/` directory
2. Place `image.png` and `meta.json` (required)
3. Set `m0_mode` in meta.json (`grid_aware` or `grid_free`)
4. The presenter auto-discovers all samples via directory scan

Add only charts you have the right to redistribute.

## Ground Truth Format

Ground truth for these samples is stored in `meta.json`, including the
question, gold answer, answer validation rule, and any curated facts used by the
presenter path. These assets are suitable for task-level presenter diagnostics,
not for point-level extrema validation.

## Licence and Attribution

These images are third-party figures and are **not** covered by the
repository's MIT licence. Each is distributed under the licence of its source
article, Creative Commons Attribution 4.0 International
([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)); the licence of each
article was confirmed against its Crossref record. Figures are reproduced as
published, cropped to a single panel.

| Sample | Source article (*Nature Communications*) | DOI |
|--------|------------------------------------------|-----|
| omniscience_002 | Structural diversity in binary superlattices self-assembled from polymer-grafted nanocrystals | [10.1038/ncomms10052](https://doi.org/10.1038/ncomms10052) |
| omniscience_003 | Tracing the fate of carbon and the atmospheric evolution of Mars | [10.1038/ncomms10003](https://doi.org/10.1038/ncomms10003) |
| omniscience_004 | The effect of soil-borne pathogens depends on the abundance of host tree species | [10.1038/ncomms10017](https://doi.org/10.1038/ncomms10017) |
| omniscience_006 | A small-gap electrostatic micro-actuator for large deflections | [10.1038/ncomms10078](https://doi.org/10.1038/ncomms10078) |
| omniscience_007 | Jurassic climate mode governed by ocean gateway | [10.1038/ncomms10015](https://doi.org/10.1038/ncomms10015) |
| omniscience_008 | Atlantic multi-decadal oscillation covaries with Agulhas leakage | [10.1038/ncomms10082](https://doi.org/10.1038/ncomms10082) |
| omniscience_009 | High-efficiency polymer solar cells with small photon energy loss | [10.1038/ncomms10085](https://doi.org/10.1038/ncomms10085) |
