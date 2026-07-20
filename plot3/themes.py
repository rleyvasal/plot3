"""Palettes and theme tokens."""

from __future__ import annotations

# Validated: light on #fcfcfb, dark on #0b1020 — dataviz reference
_CAT_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
              "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_CAT_DARK = ["#3987e5", "#199e70", "#c98500", "#008300",
             "#9085e9", "#e66767", "#d55181", "#d95926"]
# Sequential blue ramp, steps 100..700 (light -> dark). Light mode uses it as
# given (low=light); dark mode reversed so "near zero" recedes to the surface.
_SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]

# Perceptually-ordered scientific colormaps (exact matplotlib 16-stop hex) —
# multi-hue but monotonic in lightness, CVD-safe; the lidar/heatmap choice.
# Same stops in both themes (standard practice for viridis-family maps).
_VIRIDIS = ["#440154", "#481a6c", "#472f7d", "#414487", "#39568c", "#31688e",
            "#2a788e", "#23888e", "#1f988b", "#22a884", "#35b779", "#54c568",
            "#7ad151", "#a5db36", "#d2e21b", "#fde725"]
_MAGMA = ["#000004", "#0b0924", "#20114b", "#3b0f70", "#57157e", "#721f81",
          "#8c2981", "#a8327d", "#c43c75", "#de4968", "#f1605d", "#fa7f5e",
          "#fe9f6d", "#febf84", "#fddea0", "#fcfdbf"]
_TURBO = ["#30123b", "#4143a7", "#4771e9", "#3e9bfe", "#22c5e2", "#1ae4b6",
          "#46f884", "#88ff4e", "#b9f635", "#e1dd37", "#faba39", "#fd8d27",
          "#f05b12", "#d63506", "#af1801", "#7a0403"]
_CONT_PALETTES = {"viridis": _VIRIDIS, "magma": _MAGMA, "turbo": _TURBO}

_THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", cat=_CAT_LIGHT, seq=_SEQ,
    ),
    "dark": dict(
        surface="#0b1020", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#1c2742", axis="#2e3a5c", cat=_CAT_DARK,
        seq=list(reversed(_SEQ)),
    ),
}

