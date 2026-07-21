# Local testing for plot3

This folder is for **machine-local** checks (smoke scripts, HTML artifacts).
Automated pytest lives in `tests/` one level up.

## Quick start

From the plot3 repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# full unit suite (no R required)
pytest -q

# write sample HTML figures under tests/local/output/
python tests/local/smoke_local.py
```

Open any `tests/local/output/*.html` in a browser to visually check 2D/3D.

## Optional R semantic oracle

Some stats (boxplot hinges, density, histogram) can be cross-checked against
base R when `Rscript` is available:

```bash
# install R + jsonlite, then:
pytest -q tests/test_r_oracle_parity.py
```

If R is missing, those tests **skip** automatically (same pattern as tidy3).

Environment override (Pixi):

```bash
export PLOT3_R_ORACLE_MANIFEST=/path/to/pixi.toml
pytest -q tests/test_r_oracle_parity.py
```

## What is covered

| Suite | Purpose |
|-------|---------|
| `test_api_public.py` | Public exports |
| `test_grammar.py` | ggplot / aes / pipe / themes |
| `test_geoms_ds.py` | 2D DS geoms (bar, hist, box, density, violin, facet) |
| `test_3d.py` | 3D points, surface, isosurface, coord_3d |
| `test_stats_semantic.py` | Tukey / density / hist semantics without R |
| `test_r_oracle_parity.py` | Optional R oracle |
| `test_build_contract.py` | Viewer wire-format contracts |
| `local/smoke_local.py` | End-to-end HTML smoke for humans |

## Notes on “ggplot2 parity”

plot3 targets **grammar and statistical conventions** (Tukey boxplots, KDE
shape, histogram counts), not pixel-identical renders. The HTML/WebGL viewer is
checked for structure (`build_spec` contracts + smoke HTML), not against
ggplot2 PNG hashes.
