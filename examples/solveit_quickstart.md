# SolveIt quickstart — tidy3 + plot3

## 1. Load (first cell, under `%local`)

```text
%local
%run /Users/admin/gpudev/CRAFT.py
%run /Users/admin/gpudev/addons/tidy3.py
%run /Users/admin/gpudev/addons/plot3.py
```

Expect:

```text
CRAFT: tidy3 … loaded (local) …
CRAFT: plot3 … loaded (local) …
plot3 ready   # or quiet register from the addon
```

## 2. Wrangle + plot (iframe in SolveIt)

**Primary API** — same grammar everywhere (local and `%gpu`):

```python
import numpy as np

rng = np.random.default_rng(0)
# Start as a TidyFrame (tidy accepts dict / polars / pandas — no tidy(x) = …)
cars = tidy({
    "cyl": [4]*40 + [6]*40 + [8]*40,
    "mpg": list(rng.normal(26, 2.5, 40)) + list(rng.normal(20, 2.5, 40)) + list(rng.normal(15, 2.5, 40)),
    "wt":  list(rng.normal(2.2, 0.3, 40)) + list(rng.normal(3.0, 0.3, 40)) + list(rng.normal(3.8, 0.3, 40)),
    "hp":  list(rng.normal(80, 15, 40)) + list(rng.normal(110, 20, 40)) + list(rng.normal(150, 25, 40)),
})

cars
>> filter(col("hp") < 250)
>> select("wt", "mpg", "cyl")
>> ggplot(aes(x=wt, y=mpg, colour=cyl))  # bare names; use `backticks` if spaces
+ geom_point(size=5, alpha=0.85)
+ labs(title="Weight vs MPG")
+ theme_light()
```
You should see an interactive figure in the cell output. The cell gets the
**red eye** (skipped from AI context).

NumPy / point clouds (column indices):

```python
# points: (n, 4) array → x, y, z, colour
ggplot(points, aes(x=0, y=1, z=2, colour=3)) + geom_point3d()
```

## 3. Remote data (`%gpu`) — same ggplot API

```text
%gpu
```

```python
# Same syntax as local — plot3 is seeded onto the remote kernel
scan_parquet("/home/gpudev/data/example.parquet")
>> filter(col("value") > 0)
>> select("x", "y", "group")
>> ggplot(aes(x="x", y="y", colour="group"))
+ geom_point()

# Arrays on the GPU machine
ggplot(points, aes(x=0, y=1, z=2, colour=3)) + geom_point3d()
```

If remote ggplot fails after a kernel restart:

```python
seed_tidy3_remote(force=True)
seed_plot3_remote(force=True)
```

## 4. Optional host-only helper (not the main API)

Prefer section 2–3. If you stay on the **host** and only want a one-liner that
runs a remote ggplot for you, `%plot3` still exists as a thin CRAFT helper:

```python
# Host cell with CRAFT connected — not required if you use ggplot under %gpu
%plot3 pdf x=wt y=mpg color=cyl
```

## 5. VS Code vs SolveIt display

| Host | Default plot3 display |
|------|------------------------|
| SolveIt | Inline **iframe** + red-eye hide |
| VS Code notebook | System **browser** (webview blocks CDN WebGL) |

Force either: `export PLOT3_DISPLAY=iframe` or `browser`.
