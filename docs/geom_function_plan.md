# geom_function plan (plot3)

Plot mathematical functions and equations with the same grammar as data layers:
`ggplot() + geom_function("y = 2x + 2")`.

## Current state (what matters for this feature)

- **Grammar:** `ggplot(data, aes(...)) + geom_*()`; `ggplot.__add__` collects layers, labs, theme, coord, facet (`plot3/ggplot.py`).
- **Geoms** are thin param holders (`plot3/geoms.py`); each has a `kind`.
- **Stats** live in `expand_stat_geom` (`plot3/build.py:285`): bar/histogram/density compute a small pandas frame and return a drawable `geom_col`/`geom_line` with `data_override`. This is the natural hook for `geom_function`.
- **2D vs 3D** is decided in `build_spec` (`plot3/build.py:724`): 3D if any layer maps `z`. `geom_surface` draws a z-height over a complete x–y grid.
- **Data is mandatory:** `build_spec` raises if `g.data is None` (`build.py:678`). Must relax this for function-only plots.
- **Notebook bare names:** an AST transformer (`plot3/masking.py`) already rewrites bare names inside `aes()`/`facet_wrap()` into strings. The same mechanism captures `geom_function(y = 2*x + 2)`.

## Input forms

The **quoted string is the main form**. It works everywhere (scripts and notebooks) and accepts
math-style notation. Every other form is converted to the same string internally, so there is
one parser and one validator.

| Form | Where it works | Notation |
|---|---|---|
| `geom_function("y = 2x + 2")` | everywhere | math style: `2x`, `x^2`, `=` |
| `geom_function(y = 2*x + 2)` | notebooks only (AST transformer) | valid Python syntax; `^` is read as power; `2x` not allowed |
| `geom_function(lambda x: my_f(x))` | everywhere | any Python logic |

Why quotes: Python evaluates arguments before `geom_function` runs, so `y = 2x` is a
`SyntaxError`, `y = 2*x` is a `NameError`, or (after `x = 5`) silently becomes the number `10`.
A string keeps the formula intact. The notebook transformer is the only quote-free form.

## Examples

### 1. Explicit 2D curves (`y = f(x)`)

```python
ggplot() + geom_function("y = 2x")              # line through the origin
ggplot() + geom_function("y = 2x + 2")
ggplot() + geom_function("y = x^2 + 1")         # ^ means power
ggplot() + geom_function("y = 3(x - 1)^2")      # implicit * before (
ggplot() + geom_function("y = sin(x) / x")
ggplot() + geom_function("x^2 + 1")             # no left side: assumed y = ...
ggplot() + geom_function("y = 4")               # constant: horizontal line
```

### 2. Implicit equations (variable on both sides, or no lone left side)

Drawn as the curve where `F = lhs - rhs = 0` (numeric contour, no symbolic solving).

```python
ggplot() + geom_function("y = 2x + 2y")         # simplifies to y = -2x, drawn as a line
ggplot() + geom_function("x^2 + y^2 = 1")       # unit circle
ggplot() + geom_function("x y = 1")             # hyperbola
ggplot() + geom_function("x = y^2")             # sideways parabola
```

### 3. 3D surfaces (`z = f(x, y)`)

```python
ggplot() + geom_function("z = 2x + 2y")         # plane
ggplot() + geom_function("z = sin(x) cos(y)")
ggplot() + geom_function("z = x^2 - y^2", xlim=(-2, 2), ylim=(-2, 2))
ggplot() + geom_function("sin(x) cos(y)")       # two free variables, no left side: z = ...
```

### 4. Parameters assigned at the end of the call

Any name that isn't a plot variable or a math function gets its value from a keyword argument
placed after the expression.

```python
ggplot() + geom_function("y = a x^2 + b x + c", a=2, b=-3, c=1)
ggplot() + geom_function("y = A sin(k x + phi)", A=3, k=2, phi=pi/4)
ggplot() + geom_function("z = a x^2 + b y^2", a=1, b=-1)

# reusing notebook/script values: pass them explicitly
a = 2.5
ggplot() + geom_function("y = a x^2", a=a)

# several curves from one formula
p = ggplot()
for a in [0.5, 1, 2]:
    p = p + geom_function("y = a x^2", a=a)
```

### 5. User-defined functions (callable keyword arguments)

```python
def damp(t):
    return np.exp(-t / 5)

ggplot() + geom_function("y = damp(x) sin(3x)", damp=damp)
ggplot() + geom_function("y = f(x) + 1", f=my_model.predict)
```

If the function does not accept NumPy arrays, plot3 falls back to `np.vectorize` with a
one-time warning that it will be slower.

### 6. Own variable names

Free names with no value become plot variables and axis labels.

```python
ggplot() + geom_function("v = 9.8 t")           # x axis labelled t, y axis labelled v
ggplot() + geom_function("r = 1 + cos(theta)")  # x axis labelled theta
```

### 7. Notebook, no quotes (Python syntax)

```python
ggplot() + geom_function(y = 2*x + 2)
ggplot() + geom_function(y = x^2 + 1)           # ^ read as power
ggplot() + geom_function(y = a*x**2 + b, a=2, b=1)   # parameters still at the end
ggplot() + geom_function(z = sin(x)*cos(y))
```

The transformer rewrites only the expression argument (first positional, or `y=`/`z=`/`f=`).
It never touches `a=`, `xlim=`, `n=`, etc., so `xlim=(0, x_max)` evaluates normally.

### 8. Lambda (full Python logic)

```python
ggplot() + geom_function(lambda x: np.where(x < 0, 0, x**2))
ggplot() + geom_function(lambda x, y: my_f(x, y))       # 2 parameters -> 3D surface
```

### 9. Overlay on data

```python
ggplot(df, aes(x=wt, y=mpg)) + geom_point() + geom_function("y = 37 - 5x")
ggplot(df, aes(x=wt, y=mpg)) + geom_point() + geom_function("y = a + b x", a=37, b=-5)
```

The function's x domain comes from the other layers' data range.

### 10. Styling and limits

```python
ggplot() + geom_function("y = tan(x)", xlim=(-3, 3), ylim=(-10, 10))
ggplot() + geom_function("y = log(x)", color="firebrick", linewidth=2, n=1001)
ggplot() + geom_function("z = x y", xlim=(-1, 1), ylim=(-1, 1), zlim=(-0.5, 0.5), wireframe=True)
```

### 11. Helpful errors

| Input | Message |
|---|---|
| `"y = a x^2"` (no `a=`) | `'a' has no value. Pass it at the end: geom_function(..., a=2)` |
| same, with `a = 2.5` in the notebook | `'a' has no value. Your notebook has a = 2.5. Use geom_function(..., a=a)` |
| `"y = 2xy"` | `unknown name 'xy': did you mean x*y?` |
| `"y = foo(x)"` | `unknown function 'foo': pass it as geom_function(..., foo=foo)` |
| `"y = x + t + s"` | `too many free variables (x, t, s): at most 2. Pass parameters as keywords, e.g. s=1` |
| `"y = 2x +"` | `syntax error at '+' (column 7)` |

## Parsing (no sympy required)

Python's `ast.parse` does the heavy lifting. plot3 adds four small pieces in `plot3/expr.py`:

1. **Normaliser** (`tokenize`-based, not regex): `^` → `**`; implicit `*` for
   number→name (`2x`), number→`(` (`2(x+1)`), `)`→`(`, `)`→name, name→`(` when the name is
   not a function; split on a single `=` into left and right sides.
   Run-together letters are **never split**: `xy` stays one name (see the hint above).
2. **Parse:** `ast.parse(lhs)`, `ast.parse(rhs)`.
3. **Validate (security):** walk the tree and allow only `Constant`, `Name`, `BinOp`,
   `UnaryOp`, and `Call` to a known math function or a user-supplied callable. Reject
   `Attribute`, `Subscript`, comprehensions, lambdas, etc. (an empty-builtins `eval` alone is
   escapable).
4. **Classify:**
   - left side is a lone name that does not appear on the right → **explicit** (`y = f(x)`, `z = f(x, y)`, `x = f(y)`)
   - no left side → explicit, dependent variable is `y` (1 free var) or `z` (2 free vars)
   - otherwise → **implicit**, `F = lhs - rhs`, drawn where `F = 0`

Evaluation: compile once, evaluate vectorised over a grid with a namespace of NumPy functions,
numeric parameters and user callables.

## Names

- **Plot variables:** free names with no value. `x`, `y` are preferred; other names
  (`t`, `theta`) are allowed and become axis labels.
- **Math names** map to NumPy: `sin cos tan asin acos atan arcsin arccos arctan sinh cosh tanh
  exp log ln log10 sqrt abs floor ceil pi e`. A keyword argument may override one (`e=0.3`).
- **Parameters:** numeric keyword arguments at the end of the call (`a=2`).
- **User functions:** callable keyword arguments (`damp=damp`).
- **Notebook variables are never read silently.** A notebook `x = 5` never leaks in. The
  caller's namespace is inspected only to write a better error message.

## Auto 2D / 3D

| Free variables | Result |
|---|---|
| 0 | constant, horizontal line (2D) |
| 1, explicit | 2D curve → `geom_line` with `data_override` |
| 2, explicit | 3D surface → `geom_surface` over an n × n grid |
| 2, implicit | 2D contour of `F = 0` (marching squares) |
| 3, implicit | later: implicit surface `F(x, y, z) = 0` (marching cubes) |
| more | clear error |

## Limits and sampling

Limits use axis names: `xlim`, `ylim`, `zlim`. On a curve, `ylim` clips the view. On a surface,
`xlim`/`ylim` set the domain and `zlim` clips the view.

**Domain** (x, and y for surfaces and implicit curves), in priority order:

1. explicit `xlim=` / `ylim=`
2. range of other layers' data (ggplot2 behaviour, good for overlays)
3. default `(-10, 10)`; auto-narrow when the function is undefined over most of it
   (`log(x)`, `sqrt(x)` start where finite)

**Range** (y for curves, z for surfaces): non-finite values dropped; when values blow up
(`1/x`, `tan(x)`) clip to a robust percentile window, warn once that clipping happened (override
with `ylim`/`zlim`), and break the line at huge jumps instead of drawing a vertical spike.

**Sampling:** 2D `n=501`; 3D and implicit `n=80` per axis (6,400 points). Adaptive refinement
near high curvature is a later step.

**Labels:** the curve is labelled with its expression (e.g. `y = 2x + 2`) for the legend.

## Steps (each one small, reviewed before the next)

1. `plot3/expr.py`: normaliser, parser, validator, classifier, free-variable and parameter
   resolution, error messages; with tests.
2. `geom_function` 2D explicit curve with explicit `xlim`; allow `ggplot()` with no data.
3. Auto limits from other layers, default domain, y-clipping and line breaks.
4. 3D explicit surface for 2 free variables.
5. Implicit 2D contour of `F = 0`.
6. Notebook transformer in `masking.py` for `geom_function` (expression argument only).
7. Optional sympy duck-typing (`.free_symbols` + `lambdify`); README examples.

## Later ideas

- Parametric curves: `geom_function(x="cos(t)", y="sin(t)")`.
- Implicit 3D surfaces with marching cubes.
- A list of expressions in one call: `geom_function(["y = x", "y = x^2"])`.
