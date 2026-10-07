"""Turn a LaTeX formula subset into plot3's plain formula string.

The string then goes through the same normaliser and whitelist as typed
math. LaTeX does not define the meaning, so a few conventions are fixed
here: ``\\sin x^2`` is ``sin(x**2)``, ``\\sin^2 x`` is ``(sin x)**2``,
``\\sin^{-1}`` is ``arcsin``, and ``e^{x}`` is ``exp(x)``. Letters written
side by side are separate variables (``xy`` is ``x*y``). A call ``f(x)``
is left for the normaliser: it stays a call only when ``f`` was passed in.
"""

from __future__ import annotations

_EATEN_MSG = (
    'This looks like LaTeX with backslashes eaten by Python. '
    'Use a raw string: r"\\frac{1}{x}".'
)

# Python already turned the backslash into a control character.
# Form feed, bell, backspace, and vertical tab are never indentation.
_ALWAYS_EATEN = set("\a\b\f\v")
# Tab, newline, and carriage return also start real commands (\theta, \nu, \right).
# A suffix match keeps an ordinary line break such as a triple-quoted "y = x".
_EATEN_SUFFIX = {
    "\t": ("heta", "imes", "frac", "riangle", "ext", "au", "an", "o"),
    "\n": ("abla", "otin", "eq", "u", "e"),
    "\r": ("ight", "angle", "ho"),
}

_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
    "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho", "sigma": "sigma",
    "tau": "tau", "phi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
}

# Command -> the name the plain parser already knows. \ln is natural log.
_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan",
    "ln": "log", "log": "log", "exp": "exp",
}
_INVERSE = {"sin": "arcsin", "cos": "arccos", "tan": "arctan"}
_SPACING = {",", ";", ":", "!", " ", "quad", "qquad", "\\", "thinspace", "medspace", "thickspace"}
_VALUE_CMDS = {
    "frac", "dfrac", "tfrac", "sqrt", "left", "lvert", "vert", "|",
    "lfloor", "lceil", "operatorname", "mathrm", "{",
}


def reject_eaten_backslashes(source: str) -> None:
    """Raise when a normal Python string has already eaten ``\\frac`` or ``\\theta``."""
    for index, char in enumerate(source):
        if char in _ALWAYS_EATEN:
            raise ValueError(_EATEN_MSG)
        suffixes = _EATEN_SUFFIX.get(char)
        if not suffixes:
            continue
        rest = source[index + 1 :]
        for suffix in suffixes:
            if not rest.startswith(suffix):
                continue
            after = rest[len(suffix) : len(suffix) + 1]
            if after.isalpha():
                continue
            # "\ny = x" is a line break. "\nu" is a newline plus the letter u.
            if char == "\n" and (index == 0 or source[index - 1] == "\n"):
                if source.strip() != suffix:
                    continue
            raise ValueError(_EATEN_MSG)


def is_latex(text: str) -> bool:
    """True for ``$...$`` or any backslash command. Plain ``x^{2}`` is not."""
    body = text.strip()
    if "\\" in body:
        return True
    return len(body) >= 2 and body[0] == "$" and body[-1] == "$"


def unwrap_latex(text: str) -> str:
    """Drop one layer of ``$`` or ``$$`` wrappers."""
    body = text.strip()
    if body.startswith("$$") and body.endswith("$$") and len(body) >= 4:
        return body[2:-2].strip()
    if len(body) >= 2 and body[0] == "$" and body[-1] == "$":
        return body[1:-1].strip()
    return body


def latex_to_source(source: str, *, e_is_constant: bool = True) -> str:
    """Translate a LaTeX subset into ``2*x``, ``sin(x)``, ``x**(2)``."""
    parser = _Parser(_tokenize(source), e_is_constant=e_is_constant)
    return parser.parse()


def _tokenize(source: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        if char.isspace() or char == "~":
            index += 1
            continue
        if char == "\\":
            name, index = _command_at(source, index)
            if name in _SPACING:
                continue
            tokens.append(("cmd", name))
            continue
        if char.isdigit() or (char == "." and index + 1 < length and source[index + 1].isdigit()):
            end = index + 1
            seen_dot = char == "."
            while end < length and (source[end].isdigit() or (source[end] == "." and not seen_dot)):
                if source[end] == ".":
                    seen_dot = True
                end += 1
            tokens.append(("num", source[index:end]))
            index = end
            continue
        if char.isalpha():
            # One letter per variable. \theta and \operatorname carry longer names.
            tokens.append(("name", char))
            index += 1
            continue
        if source.startswith("//", index):
            tokens.append(("op", "//"))
            index += 2
            continue
        if char in "+-*/^=_,()[]{}|":
            tokens.append(("op", char))
            index += 1
            continue
        raise ValueError(f"syntax error at {char!r}")
    return tokens


def _command_at(source: str, index: int) -> tuple[str, int]:
    cursor = index + 1
    if cursor >= len(source):
        raise ValueError("syntax error at '\\\\'")
    if not source[cursor].isalpha():
        return source[cursor], cursor + 1
    end = cursor + 1
    while end < len(source) and source[end].isalpha():
        end += 1
    return source[cursor:end], end


def _neg_one(text: str) -> bool:
    return text.replace(" ", "") in {"-1", "-(1)", "(-1)"}


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], *, e_is_constant: bool):
        self.tokens = tokens
        self.index = 0
        self.e_is_constant = e_is_constant
        self.bar_depth = 0

    def parse(self) -> str:
        if not self.tokens:
            raise ValueError("syntax error at end of formula")
        left = self._expr()
        if self._eat_op("="):
            right = self._expr()
            self._finish()
            return f"{left} = {right}"
        self._finish()
        return left

    def _finish(self) -> None:
        tok = self._peek()
        if tok is not None:
            self._bad(tok)

    def _peek(self) -> tuple[str, str] | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _eat_op(self, text: str) -> bool:
        if self._peek() == ("op", text):
            self.index += 1
            return True
        return False

    def _eat_cmd(self, name: str) -> bool:
        if self._peek() == ("cmd", name):
            self.index += 1
            return True
        return False

    def _expect_op(self, text: str) -> None:
        if not self._eat_op(text):
            tok = self._peek()
            if tok is None:
                raise ValueError(f"syntax error at end of formula, expected {text!r}")
            self._bad(tok)

    def _bad(self, tok: tuple[str, str]) -> None:
        kind, val = tok
        if kind == "cmd":
            raise ValueError(f"\\{val} isn't supported in geom_function")
        raise ValueError(f"syntax error at {val!r}")

    def _starts(self, *, functions: bool) -> bool:
        tok = self._peek()
        if tok is None:
            return False
        kind, val = tok
        if kind in {"num", "name"}:
            return True
        if kind == "op":
            if val in "({[":
                return True
            return val == "|" and self.bar_depth == 0
        if kind != "cmd":
            return False
        if val in _FUNCS or val in {"operatorname", "mathrm"}:
            return functions
        if val in _VALUE_CMDS or val in _GREEK:
            return True
        # An unknown command is a factor so the error can name it.
        if val in {"cdot", "times", "bmod", "right", "rvert", "rfloor", "rceil", "end"}:
            return False
        return True

    def _expr(self) -> str:
        left = self._term()
        while True:
            if self._eat_op("+"):
                left = f"({left})+({self._term()})"
            elif self._eat_op("-"):
                left = f"({left})-({self._term()})"
            else:
                break
        return left

    def _term(self) -> str:
        left = self._unary()
        while True:
            if self._eat_op("*") or self._eat_cmd("cdot") or self._eat_cmd("times"):
                left = f"({left})*({self._unary()})"
            elif self._eat_op("/"):
                left = f"({left})/({self._unary()})"
            elif self._eat_op("//"):
                left = f"({left})//({self._unary()})"
            elif self._eat_op("%") or self._eat_cmd("bmod"):
                left = f"({left})%({self._unary()})"
            elif self._starts(functions=True):
                left = f"({left})*({self._unary()})"
            else:
                break
        return left

    def _unary(self) -> str:
        if self._eat_op("+"):
            return self._unary()
        if self._eat_op("-"):
            return f"-({self._unary()})"
        return self._power()

    def _power(self) -> str:
        base = self._atom()
        if not self._eat_op("^"):
            return base
        exp = self._script()
        if base == "e" and self.e_is_constant:
            return f"exp({exp})"
        return f"({base})**({exp})"

    def _script(self) -> str:
        if self._eat_op("{"):
            inner = self._expr()
            self._expect_op("}")
            return inner
        return self._unary()

    def _atom(self) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("syntax error at end of formula")
        kind, val = tok
        if kind == "num":
            self.index += 1
            return val
        if kind == "name":
            self.index += 1
            base = self._subscript(val)
            if self._peek() == ("op", "("):
                return self._call(base)
            return base
        if kind == "op":
            if val == "(":
                return self._group(")")
            if val == "[":
                return self._group("]")
            if val == "{":
                return self._group("}")
            if val == "|":
                return self._bars()
            self._bad(tok)
        if val in _FUNCS:
            return self._func(val)
        if val in {"frac", "dfrac", "tfrac"}:
            return self._frac()
        if val == "sqrt":
            return self._sqrt()
        if val in {"operatorname", "mathrm"}:
            return self._word()
        if val in _GREEK:
            self.index += 1
            return self._subscript(_GREEK[val])
        if val == "left":
            return self._left()
        if val in {"lvert", "vert", "|"}:
            return self._abs_until({"rvert", "vert", "|"})
        if val == "lfloor":
            return self._abs_until({"rfloor"}, wrapper="floor")
        if val == "lceil":
            return self._abs_until({"rceil"}, wrapper="ceil")
        if val == "{":
            return self._group("}")
        self._bad(tok)

    def _group(self, closer: str) -> str:
        self.index += 1
        inner = self._expr()
        if closer == "}" and self._eat_cmd("}"):
            return f"({inner})"
        self._expect_op(closer)
        return f"({inner})"

    def _bars(self) -> str:
        self.index += 1
        self.bar_depth += 1
        try:
            inner = self._expr()
        finally:
            self.bar_depth -= 1
        self._expect_op("|")
        return f"abs({inner})"

    def _subscript(self, base: str) -> str:
        if not self._eat_op("_"):
            return base
        if self._eat_op("{"):
            sub = self._expr()
            self._expect_op("}")
        else:
            sub = self._script_atom()
        if not sub or any(not (ch.isalnum() or ch == "_") for ch in sub):
            raise ValueError(f"syntax error at subscript {sub!r}")
        return f"{base}_{sub}"

    def _script_atom(self) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("syntax error at end of formula")
        kind, val = tok
        if kind in {"num", "name"}:
            self.index += 1
            return val
        if kind == "cmd" and val in _GREEK:
            self.index += 1
            return _GREEK[val]
        self._bad(tok)

    def _call(self, name: str) -> str:
        self._expect_op("(")
        args: list[str] = []
        if self._peek() != ("op", ")"):
            args.append(self._expr())
            while self._eat_op(","):
                args.append(self._expr())
        self._expect_op(")")
        return f"{name}({', '.join(args)})"

    def _func(self, cmd: str) -> str:
        self.index += 1
        name = _FUNCS[cmd]
        power = None
        inverse = False
        for _ in range(2):
            if power is None and not inverse and self._eat_op("^"):
                exp = self._script()
                if _neg_one(exp):
                    inverse = True
                else:
                    power = exp
            elif cmd == "log" and name == "log" and self._eat_op("_"):
                sub = self._script()
                if sub != "10":
                    raise ValueError(
                        "\\log with a subscript other than 10 isn't supported "
                        "in geom_function"
                    )
                name = "log10"
            else:
                break
        if inverse:
            mapped = _INVERSE.get(name)
            if mapped:
                name = mapped
            else:
                power = "-(1)"
        if self._eat_op("{"):
            arg = self._expr()
            self._expect_op("}")
        elif self._peek_sign() or self._starts(functions=True):
            arg = self._unary()
            while self._starts(functions=False):
                arg = f"({arg})*({self._unary()})"
        else:
            raise ValueError(f"syntax error at '\\{cmd}', expected an argument")
        call = f"{name}({arg})"
        if power is not None:
            return f"({call})**({power})"
        return call

    def _peek_sign(self) -> bool:
        tok = self._peek()
        return tok is not None and tok[0] == "op" and tok[1] in "+-"

    def _frac(self) -> str:
        self.index += 1
        num = self._braced()
        den = self._braced()
        return f"({num})/({den})"

    def _braced(self) -> str:
        if not self._eat_op("{"):
            tok = self._peek()
            if tok is None:
                raise ValueError("syntax error at end of formula")
            self._bad(tok)
        inner = self._expr()
        self._expect_op("}")
        return inner

    def _sqrt(self) -> str:
        self.index += 1
        index = None
        if self._eat_op("["):
            index = self._expr()
            self._expect_op("]")
        body = self._braced()
        if index is None or index in {"2", "(2)"}:
            return f"sqrt({body})"
        if index in {"3", "(3)"}:
            return f"cbrt({body})"
        return f"({body})**(1/({index}))"

    def _word(self) -> str:
        self.index += 1
        if not self._eat_op("{"):
            raise ValueError("syntax error at '\\mathrm', expected a name")
        parts: list[str] = []
        while self._peek() not in {None, ("op", "}")}:
            tok = self._peek()
            assert tok is not None
            kind, val = tok
            if kind in {"name", "num"} or (kind == "op" and val == "_"):
                parts.append(val)
                self.index += 1
                continue
            self._bad(tok)
        self._expect_op("}")
        word = "".join(parts)
        if not word.isidentifier():
            raise ValueError(f"syntax error at '\\mathrm{{{word}}}'")
        if self._peek() == ("op", "("):
            return self._call(word)
        return word

    def _left(self) -> str:
        self.index += 1
        tok = self._peek()
        if tok is None:
            raise ValueError("syntax error at '\\left'")
        kind, val = tok
        self.index += 1
        if val in {"|", "lvert", "vert"}:
            inner = self._expr()
            self._right({"|", "rvert", "vert"})
            return f"abs({inner})"
        if val in {"(", "[", "{", "lbrace"}:
            inner = self._expr()
            self._right({")", "]", "}", "rbrace"})
            return f"({inner})"
        if kind == "cmd":
            raise ValueError(f"\\{val} isn't supported in geom_function")
        raise ValueError(f"syntax error at {val!r}")

    def _right(self, closers: set[str]) -> None:
        if not self._eat_cmd("right"):
            tok = self._peek()
            if tok is None:
                raise ValueError("syntax error at end of formula, expected \\right")
            self._bad(tok)
        tok = self._peek()
        if tok is None or tok[1] not in closers:
            raise ValueError("syntax error at '\\right'")
        self.index += 1

    def _abs_until(self, closers: set[str], wrapper: str = "abs") -> str:
        self.index += 1
        inner = self._expr()
        tok = self._peek()
        if tok is not None and tok[1] in closers and (tok[0] == "cmd" or tok == ("op", "|")):
            self.index += 1
            return f"{wrapper}({inner})"
        if tok is None:
            raise ValueError("syntax error at end of formula")
        self._bad(tok)
