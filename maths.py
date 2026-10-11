"""Maths for Kookie, done properly instead of hopefully.

A 4B model asked "what is 123456789 * 987654321" produces a confident wrong
number, and a confident wrong number is worse than no number: it gets believed.
So arithmetic is not left to the model at all. She is handed a tool, and this is
what runs behind it.

Two layers, because they fail differently:

- Expressions go through sympy, so they are exact. 2**100 is the real number and
  not a rounded one, sqrt(2) stays sqrt(2), and 0.1 + 0.2 is 0.30000000000000004
  because that genuinely is what those two floats add to - or the exact 3/10 when
  the question is about money, which is why decimals are handled as rationals.
- Named work - solving, differentiating, integrating, factoring, statistics,
  number theory, series, unit conversion, interest - is done by a command table,
  because "integrate x^2" is not an expression and there is no way to guess that
  from text.

Everything is whitelisted. sympify evaluates a string, so a string that could
reach the interpreter is a string that must never arrive: attribute access,
dunder names, imports and lambdas are refused before anything is parsed, and the
namespace handed to the parser holds only the names listed here.

The sympy import is deliberately inside the functions. It is about a second and
tens of megabytes to load, and this process is a 4B model on a 7.6 GB box that
is usually memory starved - importing it at module load would tax every start for
a tool used once in a while. The pure-Python layer underneath means she still does
arithmetic, statistics and number theory correctly if sympy is ever missing.
"""

import ast
import math
import operator
import re
import statistics

MAX_ARGUMENT_CHARS = 4000
# Long enough for a big factorial or a long sum, short enough that a pasted
# paragraph is refused instead of parsed.
MAX_EXPRESSION_CHARS = 600
MAX_OUTPUT_CHARS = 900
MAX_LIST_ITEMS = 5000

# Anything that could turn an expression into code. Checked before parsing, on the
# raw text, because the point is to never hand the interpreter a string that
# mentions these at all.
_FORBIDDEN = (
    "__", "import", "lambda", "eval", "exec", "open", "compile", "globals",
    "locals", "getattr", "setattr", "delattr", "breakpoint", ";", "`", "\\",
)


class MathError(Exception):
    """Something she should say out loud rather than a traceback."""


def _lazy_sympy():
    try:
        import sympy
    except ImportError as error:  # pragma: no cover - depends on the install
        raise MathError(
            "The proper maths library is not installed, so only plain arithmetic, "
            "statistics and number theory are available right now."
        ) from error
    return sympy


# ---------------------------------------------------------------- pure python

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "int": int, "float": float, "pow": pow, "sqrt": math.sqrt,
    "exp": math.exp, "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin,
    "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "degrees": math.degrees, "radians": math.radians,
    "factorial": math.factorial, "gcd": math.gcd, "comb": math.comb,
    "perm": math.perm, "hypot": math.hypot, "floor": math.floor,
    "ceil": math.ceil, "trunc": math.trunc, "prod": math.prod,
    "fact": math.factorial, "choose": math.comb, "lcm": math.lcm,
    "isqrt": math.isqrt, "fabs": math.fabs, "fsum": math.fsum,
}
_CONSTANTS = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "nan": math.nan, "phi": (1 + 5 ** 0.5) / 2,
}
# Names a caller is allowed to invent, so x, n, r and friends can be given a
# value and the expression evaluated. Bounded, or this is a memory leak.
_VARIABLES = {}
MAX_VARIABLES = 40


def _eval_node(node, variables):
    """Evaluate one node of a parsed expression, refusing anything unknown."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathError("Only numbers are allowed in a calculation.")
        return node.value
    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPS.get(type(node.op))
        if handler is None:
            raise MathError("That operator is not one I can work out.")
        return handler(_eval_node(node.left, variables), _eval_node(node.right, variables))
    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise MathError("That sign is not one I can work out.")
        return handler(_eval_node(node.operand, variables))
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None)
        function = _FUNCTIONS.get(name)
        if function is None:
            raise MathError(f"I do not have a function called {name}.")
        args = [_eval_node(item, variables) for item in node.args]
        try:
            return function(*args)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as error:
            raise MathError(f"That is not a calculation I can do: {error}") from error
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise MathError(f"I have not been told what {node.id} is.")
    if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        return [_eval_node(item, variables) for item in node.elts]
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, variables)
            if isinstance(op, ast.Eq) and not left == right:
                return False
            if isinstance(op, ast.NotEq) and not left != right:
                return False
            if isinstance(op, ast.Lt) and not left < right:
                return False
            if isinstance(op, ast.LtE) and not left <= right:
                return False
            if isinstance(op, ast.Gt) and not left > right:
                return False
            if isinstance(op, ast.GtE) and not left >= right:
                return False
            left = right
        return True
    raise MathError("That is not a calculation I understand.")


def _numbers_in(text):
    """Every number in a string, as floats, for the statistics commands."""
    found = re.findall(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text or "")),
    )
    if not found:
        raise MathError("I need some numbers to work with there.")
    if len(found) > MAX_LIST_ITEMS:
        raise MathError("That is more numbers than I can hold at once.")
    return [float(item) for item in found]


def _format_number(value):
    """A number as something worth reading out loud.

    Exact where it is exact, and never in exponent form for a plain integer:
    2**100 printed as 1.2676506002282294e+30 is correct and useless.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "not a number"
        if value in (math.inf, -math.inf):
            return "infinity" if value > 0 else "minus infinity"
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        text = f"{value:.10g}"
        return text
    return str(value)


# ------------------------------------------------------------------ units

_UNITS = {
    # length, in metres
    "m": ("length", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
    "metre": ("length", 1.0), "metres": ("length", 1.0), "km": ("length", 1000.0),
    "kilometer": ("length", 1000.0), "kilometers": ("length", 1000.0),
    "kilometre": ("length", 1000.0), "kilometres": ("length", 1000.0),
    "cm": ("length", 0.01), "centimeter": ("length", 0.01),
    "centimeters": ("length", 0.01), "mm": ("length", 0.001),
    "mile": ("length", 1609.344), "miles": ("length", 1609.344),
    "mi": ("length", 1609.344), "yard": ("length", 0.9144),
    "yards": ("length", 0.9144), "ft": ("length", 0.3048),
    "foot": ("length", 0.3048), "feet": ("length", 0.3048),
    "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    "nmi": ("length", 1852.0),
    # mass, in kilograms
    "kg": ("mass", 1.0), "kilogram": ("mass", 1.0), "kilograms": ("mass", 1.0),
    "g": ("mass", 0.001), "gram": ("mass", 0.001), "grams": ("mass", 0.001),
    "mg": ("mass", 1e-6), "t": ("mass", 1000.0), "tonne": ("mass", 1000.0),
    "tonnes": ("mass", 1000.0), "lb": ("mass", 0.45359237),
    "pound": ("mass", 0.45359237), "pounds": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523), "ounce": ("mass", 0.028349523),
    "ounces": ("mass", 0.028349523), "stone": ("mass", 6.35029),
    # time, in seconds
    "s": ("time", 1.0), "sec": ("time", 1.0), "second": ("time", 1.0),
    "seconds": ("time", 1.0), "min": ("time", 60.0), "minute": ("time", 60.0),
    "minutes": ("time", 60.0), "h": ("time", 3600.0), "hr": ("time", 3600.0),
    "hour": ("time", 3600.0), "hours": ("time", 3600.0),
    "day": ("time", 86400.0), "days": ("time", 86400.0),
    "week": ("time", 604800.0), "weeks": ("time", 604800.0),
    "year": ("time", 31557600.0), "years": ("time", 31557600.0),
    # data, in bytes
    "b": ("data", 1.0), "byte": ("data", 1.0), "bytes": ("data", 1.0),
    "kb": ("data", 1000.0), "mb": ("data", 1e6), "gb": ("data", 1e9),
    "tb": ("data", 1e12), "kib": ("data", 1024.0), "mib": ("data", 1024 ** 2),
    "gib": ("data", 1024 ** 3), "tib": ("data", 1024 ** 4),
}
# Temperature is not a ratio, so it gets its own conversion rather than a factor.
_TEMPERATURES = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
_TEMPERATURE_NAMES = {
    "c": ("c", 0.0), "celsius": ("c", 0.0), "f": ("f", 32.0),
    "fahrenheit": ("f", 32.0), "k": ("k", 273.15), "kelvin": ("k", 273.15),
}


def _convert_units(amount, source, target):
    source, target = source.strip().lower(), target.strip().lower()
    if source in _TEMPERATURES and target in _TEMPERATURES:
        # Not a ratio, so it gets its own maths. 98.6F came out as 66.6C while
        # this subtracted the 32 and forgot the 9/5, which is the whole conversion.
        from_name, from_offset = _TEMPERATURE_NAMES[source]
        to_name, to_offset = _TEMPERATURE_NAMES[target]
        if from_name == "f":
            celsius = (amount - from_offset) * 5 / 9
        elif from_name == "k":
            celsius = amount - from_offset
        else:
            celsius = amount
        if to_name == "f":
            return celsius * 9 / 5 + to_offset
        if to_name == "k":
            return celsius + to_offset
        return celsius
    if source not in _UNITS or target not in _UNITS:
        raise MathError(f"I do not know how to convert {source} to {target}.")
    from_kind, from_factor = _UNITS[source]
    to_kind, to_factor = _UNITS[target]
    if from_kind != to_kind:
        raise MathError(
            f"{source} measures {from_kind} and {target} measures {to_kind}, so "
            "there is no honest way to convert between them."
        )
    return amount * from_factor / to_factor


# -------------------------------------------------------------- sympy layer

def _sympy_namespace(sympy):
    """The only names a parsed expression may see."""
    return {
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
        "atan2": sympy.atan2, "sinh": sympy.sinh, "cosh": sympy.cosh,
        "tanh": sympy.tanh, "sqrt": sympy.sqrt, "cbrt": sympy.cbrt,
        "exp": sympy.exp, "log": sympy.log, "ln": sympy.log, "log10": sympy.log,
        "Abs": sympy.Abs, "abs": sympy.Abs, "sign": sympy.sign,
        "floor": sympy.floor, "ceiling": sympy.ceiling,
        "Min": sympy.Min, "Max": sympy.Max, "min": sympy.Min, "max": sympy.Max,
        "Rational": sympy.Rational, "Integer": sympy.Integer, "Float": sympy.Float,
        "factorial": sympy.factorial, "factorial2": sympy.factorial2,
        "binomial": sympy.binomial, "comb": sympy.binomial, "choose": sympy.binomial,
        # sympy moved these around between versions, so they are looked up
        # rather than named: one missing name used to raise while the namespace
        # was built, which took every symbolic command down with it.
        "perm": getattr(sympy, "perm", sympy.factorial),
        "gcd": sympy.gcd, "lcm": sympy.lcm, "igcd": sympy.igcd,
        "isprime": sympy.isprime, "nextprime": sympy.nextprime,
        "primepi": sympy.primepi, "divisor_count": sympy.divisor_count,
        "totient": sympy.totient, "fibonacci": sympy.fibonacci,
        "Matrix": sympy.Matrix, "matrices": None, "pi": sympy.pi,
        "E": sympy.E, "I": sympy.I, "oo": sympy.oo, "inf": sympy.oo,
        "nan": sympy.nan, "e": sympy.E, "tau": sympy.pi * 2,
        "GoldenRatio": sympy.GoldenRatio, "EulerGamma": sympy.EulerGamma,
        # Lowercase i, because everyone writes (2+3i) and not (2+3I).
        "i": sympy.I, "j": sympy.I,
        "conjugate": sympy.conjugate, "re": sympy.re, "im": sympy.im,
        "arg": sympy.arg, "exp_polar": sympy.exp_polar,
    }


def _check_text(text, what="calculation"):
    text = str(text or "").strip()
    if not text:
        raise MathError("There was nothing to work out.")
    if len(text) > MAX_ARGUMENT_CHARS:
        raise MathError("That is too long to be a maths question.")
    for bad in _FORBIDDEN:
        if bad in text:
            raise MathError(
                "That contains something I am not allowed to run, so I did not run it."
            )
    return text


def _clean_expression(text):
    """Make ordinary typing into something a parser can read."""
    cleaned = str(text or "").strip()
    # "1,847.63" is one number. Only a group of exactly three digits counts, so
    # a matrix written as [1,2,3] keeps its separators.
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", cleaned)
    cleaned = cleaned.replace("^", "**")          # spreadsheets and calculators
    # "15% of 200" is a fifth of 200 and "17 % 5" is a remainder. The first is
    # rewritten whole, before the sign is looked at on its own, because a rule
    # for a bare "%" turns the first into "15 / 100 of 200" and then there is no
    # percentage left to find.
    cleaned = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
        r"(\1/100)*\2", cleaned, flags=re.IGNORECASE,
    )
    # What is left is a bare sign, and a digit after it means a modulus.
    cleaned = re.sub(r"(?<=\d)\s*%\s*(?!\s*[0-9])", "/ 100 ", cleaned)
    cleaned = re.sub(r"\s*×\s*", "*", cleaned)
    cleaned = re.sub(r"\s*÷\s*", "/", cleaned)
    cleaned = re.sub(r"\s*−\s*", "-", cleaned)     # a real minus sign
    cleaned = cleaned.replace("π", "pi").replace("√", "sqrt")
    return cleaned.strip()


def _symbols_in(text):
    return sorted(
        {name for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(text or ""))}
    )


def _format_sympy(value, digits=12):
    """A sympy result as text, with a decimal beside it when that helps."""
    try:
        if getattr(value, "is_Boolean", False):
            return str(value)
        text = str(value)
    except Exception as error:  # pragma: no cover - defensive
        raise MathError(f"I could not make sense of that result: {error}") from error
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[: MAX_OUTPUT_CHARS - 1] + "…"
    extra = ""
    try:
        numeric = complex(value.evalf(digits))
        if abs(numeric.imag) < 1e-12:
            real = numeric.real
            if abs(real - round(real)) < 1e-9 and abs(real) < 1e15:
                approximation = _format_number(int(round(real)))
            else:
                approximation = _format_number(round(real, digits))
            if approximation not in text and "I" not in text:
                extra = f"  (about {approximation})"
        else:
            approximation = _format_number(round(numeric.real, digits))
            if approximation != _format_number(round(numeric.imag, digits)):
                approximation += f" + {_format_number(round(numeric.imag, digits))}i"
            if approximation not in text:
                extra = f"  (about {approximation})"
    except (TypeError, ValueError, AttributeError, ZeroDivisionError, OverflowError):
        pass
    return text + extra


def _evaluate_sympy(expression, variables=None):
    sympy = _lazy_sympy()
    expression = _clean_expression(expression)
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise MathError("That expression is too long for me to hold in my head.")
    namespace = _sympy_namespace(sympy)
    namespace.pop("matrices", None)
    free = _symbols_in(expression)
    for name in free:
        # Anything not a known function is a variable, which is how "x + 2" is
        # answered with a formula instead of an error.
        if name not in namespace:
            namespace[name] = sympy.Symbol(name)
    # parse_expr rather than sympify, because it can be told to read "3x" and
    # "2 sqrt(2)" as the multiplication everyone writing them means. sympify has
    # no such option, and "x**2 + 3x - 4" is how a person writes that equation.
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application,
        )
        value = parse_expr(
            expression,
            local_dict=namespace,
            global_dict={"__builtins__": {}},
            transformations=standard_transformations + (implicit_multiplication_application,),
            evaluate=True,
        )
        return value, namespace
    except MathError:
        raise
    except (SyntaxError, TypeError, ValueError, AttributeError, KeyError,
            IndexError, ZeroDivisionError) as error:
        raise MathError(f"I could not work that out: {error}") from error


# ------------------------------------------------------------- command table

def _cmd_expression(rest, sympy=None):
    value, _ = _evaluate_sympy(rest)
    return _format_sympy(value)


def _cmd_simplify(rest, sympy=None):
    value, _ = _evaluate_sympy(rest)
    return _format_sympy(sympy.simplify(value))


def _cmd_expand(rest, sympy=None):
    value, _ = _evaluate_sympy(rest)
    return _format_sympy(sympy.expand(value))


def _cmd_factor(rest, sympy=None):
    # "factor x**2-5x+6" is a polynomial and "factor 360" is a whole number, and
    # they are different questions that happen to share a word. Handed the second
    # to the polynomial one it printed 360 back, which is technically a factor of
    # 360 and useless as an answer.
    plain = re.fullmatch(r"(?:factor(?:ise|ize)?|integer factor(?:isation|ization)?)?\s*([\d,]+)", str(rest or "").strip(), re.IGNORECASE)
    if plain:
        return _cmd_numbers(f"factor {plain.group(1).replace(',', '')}", sympy)
    value, _ = _evaluate_sympy(rest)
    return _format_sympy(sympy.factor(value))


def _cmd_solve(rest, sympy=None):
    text = _clean_expression(rest)
    # Two equations separated by a comma is a system, and a 4B model asked to
    # solve one writes both in a single call. Split them here instead of handing
    # sympy "(x+y) - (10, x-y=2)", which is not an expression.
    if "," in text and text.count("=") >= 2:
        equations = [item.strip() for item in text.split(",") if item.strip()]
        lines = []
        for equation in equations:
            lines.append(_cmd_solve(equation, sympy))
        return " ; ".join(lines)
    expression = text
    if "=" not in expression:
        expression = f"{expression} = 0"
    left, _, right = expression.partition("=")
    value, namespace = _evaluate_sympy(f"({left}) - ({right})")
    unknowns = sorted(
        {item for item in value.free_symbols if str(item) not in ("I",)},
        key=str,
    )
    if not unknowns:
        return "That has no variable in it, so there is nothing to solve for."
    target = unknowns[0]
    others = unknowns[1:]
    solution = sympy.solve(value, target)
    if len(others) == 1 and isinstance(solution, list) and solution:
        # A system: substitute back and report the pairs.
        pairs = []
        for first in solution:
            try:
                second = sympy.solve(value.subs(target, first), others[0])
            except (TypeError, ValueError):
                second = []
            for item in (second if isinstance(second, list) else [second]):
                pairs.append(f"{target} = {_format_sympy(first)}, {others[0]} = {_format_sympy(item)}")
        if pairs:
            return "; ".join(pairs[:6]) + (f" (and {len(pairs) - 6} more)" if len(pairs) > 6 else "")
    if not solution:
        return f"There is no value of {target} that solves that."
    if isinstance(solution, dict):
        return "; ".join(f"{key} = {_format_sympy(item)}" for key, item in solution.items())
    # Named, because "2, 3" is an answer to a question she was not asked: asked
    # to solve for x, the reply has to say which variable those numbers are.
    return ", ".join(f"{target} = {_format_sympy(item)}" for item in solution[:8])


def _wrt(expression, sympy):
    """Pull the "with respect to x" off the end of a command."""
    match = re.search(r"\b(?:wrt|with respect to|d/d|differentiate w\.r\.t\.?|derivative of|integrate over)\s+([A-Za-z_]\w*)", expression)
    if not match:
        return expression, None
    return expression[: match.start()].strip(), match.group(1)


def _cmd_diff(rest, sympy=None):
    expression, variable = _wrt(_clean_expression(rest), sympy)
    value, namespace = _evaluate_sympy(expression)
    if variable:
        target = sympy.Symbol(variable)
    elif value.free_symbols:
        target = sorted(value.free_symbols, key=str)[0]
    else:
        return "There is no variable in that to differentiate."
    return f"d/d{target} ({_format_sympy(value)}) = {_format_sympy(sympy.diff(value, target))}"


def _cmd_integrate(rest, sympy=None):
    expression = _clean_expression(rest)
    limits = re.search(r"\bfrom\s+(-?[\d.]+)\s+to\s+(-?[\d.]+)", expression)
    if limits:
        low, high = float(limits.group(1)), float(limits.group(2))
        expression = expression[: limits.start()].strip()
    else:
        low = high = None
    expression, variable = _wrt(expression, sympy)
    expression = re.sub(r"\b(?:d|dx|x)\s*$", "", expression).strip() or expression
    value, _ = _evaluate_sympy(expression)
    if variable:
        target = sympy.Symbol(variable)
    elif value.free_symbols:
        target = sorted(value.free_symbols, key=str)[0]
    else:
        return "There is no variable in that to integrate."
    if low is not None and high is not None:
        definite = sympy.integrate(value, (target, sympy.Rational(str(low)), sympy.Rational(str(high))))
        return f"The area from {low} to {high} is {_format_sympy(definite)}"
    return f"∫ {_format_sympy(value)} d{target} = {_format_sympy(sympy.integrate(value, target))} + C"


def _cmd_limit(rest, sympy=None):
    match = re.match(r"(.*?)\s*(?:as|when|at)\s+([A-Za-z_]\w*)\s*(?:->|→|goes to|tends to|approaches|→)\s*(.+)$", _clean_expression(rest), re.IGNORECASE)
    if not match:
        return "Tell me the limit like this: limit x**2 as x -> 0"
    expression, variable, point = match.group(1), match.group(2), match.group(3).strip()
    value, _ = _evaluate_sympy(expression)
    target = sympy.Symbol(variable)
    approach = _evaluate_sympy(point)[0]
    return f"lim({_format_sympy(value)}, {variable} -> {point}) = {_format_sympy(sympy.limit(value, target, approach))}"


def _cmd_series(rest, sympy=None):
    match = re.match(r"(.*?)\s*(?:around|about|near|at)\s+([A-Za-z_]\w*)\s*=\s*(-?[\d.]+)(?:\s*(?:to|order)\s*(\d+))?$", _clean_expression(rest), re.IGNORECASE)
    if not match:
        return "Tell me the expansion like this: series exp(x) around x = 0 to 6"
    expression, variable, point, order = match.group(1), match.group(2), match.group(3), match.group(4) or "6"
    value, _ = _evaluate_sympy(expression)
    target = sympy.Symbol(variable)
    expansion = sympy.series(value, target, sympy.Rational(point), int(order))
    return f"{_format_sympy(value)} about {variable} = {point} = {_format_sympy(expansion)}"


def _cmd_substitute(rest, sympy=None):
    match = re.match(r"(.*?)\s*(?:with|where|at|when)\s+([A-Za-z_]\w*)\s*=\s*(.+)$", _clean_expression(rest), re.IGNORECASE)
    if not match:
        return "Tell me what to put in like this: substitute x**2+1 with x = 3"
    expression, variable, value_text = match.group(1), match.group(2), match.group(3)
    value, _ = _evaluate_sympy(expression)
    replacement, _ = _evaluate_sympy(value_text)
    return _format_sympy(value.subs(sympy.Symbol(variable), replacement))


def _cmd_numeric(rest, sympy=None):
    """A decimal for something exact, which is what a person usually wants."""
    value, _ = _evaluate_sympy(rest)
    try:
        approximation = complex(value.evalf(15))
    except (TypeError, ValueError, AttributeError) as error:
        raise MathError(f"That does not have a decimal form: {error}") from error
    if abs(approximation.imag) < 1e-15:
        return _format_number(round(approximation.real, 12))
    return _format_number(round(approximation.real, 12)) + " + " + _format_number(round(approximation.imag, 12)) + "i"


def _cmd_matrix(rest, sympy=None):
    rows = re.findall(r"\[([^\[\]]*)\]", str(rest or ""))
    if not rows:
        return "Give me the matrix like this: matrix [[1,2],[3,4]] det"
    try:
        matrix = sympy.Matrix([[sympy.sympify(cell.strip()) for cell in row.split(",")] for row in rows])
    except (SyntaxError, TypeError, ValueError) as error:
        raise MathError(f"I could not read that matrix: {error}") from error
    if matrix.rows != matrix.cols:
        raise MathError("A determinant and an inverse need a square matrix.")
    what = (re.search(r"\b(det|determinant|inverse|inv|rank|trace|transpose|eigen|eigenvalues)\b", str(rest), re.IGNORECASE) or [None, "det"])[1].lower()
    if what in ("det", "determinant"):
        return f"det = {_format_sympy(matrix.det())}"
    if what in ("inverse", "inv"):
        return "inverse = " + str(matrix.inv())
    if what == "rank":
        return f"rank = {matrix.rank()}"
    if what == "trace":
        return f"trace = {_format_sympy(matrix.trace())}"
    if what == "transpose":
        return str(matrix.T)
    if what in ("eigen", "eigenvalues"):
        return "eigenvalues = " + str(matrix.eigenvals())
    return f"det = {_format_sympy(matrix.det())}; inverse = {matrix.inv()}"


def _cmd_statistics(rest, sympy=None):
    text = str(rest or "").strip().lower()
    values = _numbers_in(text)
    if re.search(r"\b(mean|average|avg)\b", text):
        return f"mean = {_format_number(round(statistics.fmean(values), 10))}"
    if re.search(r"\bmedian\b", text):
        return f"median = {_format_number(statistics.median(values))}"
    if re.search(r"\bmode\b", text):
        try:
            return f"mode = {_format_number(statistics.mode(values))}"
        except statistics.StatisticsError:
            return "There is no mode: every number here appears the same number of times."
    if re.search(r"\b(stdev|std|sd|standard deviation)\b", text):
        if len(values) < 2:
            return "A standard deviation needs at least two numbers."
        return f"standard deviation = {_format_number(round(statistics.stdev(values), 10))}"
    if re.search(r"\b(variance|var)\b", text):
        if len(values) < 2:
            return "A variance needs at least two numbers."
        return f"variance = {_format_number(round(statistics.variance(values), 10))}"
    if re.search(r"\b(range|spread)\b", text):
        return f"range = {_format_number(max(values) - min(values))}"
    if re.search(r"\b(sum|total)\b", text):
        return f"sum = {_format_number(round(math.fsum(values), 10))}"
    if re.search(r"\b(count|how many)\b", text):
        return f"there are {len(values)} numbers"
    return (
        f"n = {len(values)}, sum = {_format_number(round(math.fsum(values), 10))}, "
        f"mean = {_format_number(round(statistics.fmean(values), 10))}, "
        f"median = {_format_number(statistics.median(values))}, "
        f"min = {_format_number(min(values))}, max = {_format_number(max(values))}"
    )


def _cmd_numbers(rest, sympy=None):
    text = str(rest or "").strip()
    numbers = [int(float(item)) for item in re.findall(r"\d+", text)]
    if not numbers:
        raise MathError("I need at least one whole number for that.")
    first = numbers[0]
    action = text.lower()
    if re.search(r"\b(prime|factori[sz]e|factor|divisors|composite)\b", action):
        sympy_module = _lazy_sympy()
        if re.search(r"\bcomposite\b", action):
            return f"{first} is {'not prime' if sympy_module.isprime(first) else 'composite'}."
        if re.search(r"\bdivisors\b", action):
            return "divisors = " + str(sympy_module.divisors(first))
        if re.search(r"\bfactori?sz", action):
            return f"{first} = {sympy_module.factorint(first)}"
        if sympy_module.isprime(first):
            return f"{first} is prime."
        return f"{first} is not prime; it is {sympy_module.factorint(first)}"
    if len(numbers) >= 2 and re.search(r"\bgcd\b", action):
        return f"gcd = {math.gcd(numbers[0], numbers[1])}"
    if len(numbers) >= 2 and re.search(r"\blcm\b", action):
        return f"lcm = {math.lcm(numbers[0], numbers[1])}"
    if re.search(r"\b(bin|binary|oct|octal|hex|hexadecimal|base)\b", action):
        match = re.search(r"\b(?:to\s+(bin|binary|oct|octal|hex|hexadecimal|base\s*\d+)|in\s+(bin|binary|oct|octal|hex|hexadecimal))", action)
        base = {"bin": 2, "binary": 2, "oct": 8, "octal": 8, "hex": 16, "hexadecimal": 16}
        if match:
            word = match.group(1)
            target = base.get(word.split()[0], int(word.split()[1]) if "base" in word else 10)
        else:
            target = 2
        if target == 2:
            return f"{first} in binary is {first:b}"
        if target == 8:
            return f"{first} in octal is {first:o}"
        if target == 16:
            return f"{first} in hexadecimal is {first:X}"
        if 2 <= target <= 36:
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"
            number, out = first, ""
            while number:
                number, remainder = divmod(number, target)
                out = digits[remainder] + out
            return f"{first} in base {target} is {out or '0'}"
    if re.search(r"\b(factorial|fact)\b", action):
        if first > 20000:
            raise MathError("That factorial has over 77000 digits, which I will not print.")
        return f"{first}! = {math.factorial(first)}"
    if re.search(r"\b(fibonacci|fib)\b", action):
        if first > 10000:
            raise MathError("That Fibonacci number is too long to print.")
        a, b = 0, 1
        for _ in range(first):
            a, b = b, a + b
        return f"fibonacci({first}) = {a}"
    if re.search(r"\b(sqrt|square root|root)\b", action):
        root = math.isqrt(first)
        exact = root * root == first
        return f"sqrt({first}) = {root}" + ("" if exact else f" (about {first ** 0.5:.10g})")
    if re.search(r"\bpercent\b", action):
        return f"{first}% = {_format_number(first / 100)}"
    if re.search(r"\btriangular\b", action):
        return f"triangular({first}) = {first * (first + 1) // 2}"
    if re.search(r"\bsquare\b", action):
        return f"{first}^2 = {first ** 2}"
    if re.search(r"\bcube\b", action):
        return f"{first}^3 = {first ** 3}"
    return f"{first} in words: " + _number_words(first)


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _number_words(number):
    if abs(number) >= 10 ** 18:
        return f"{number:,} (too long to say in words)"
    if number < 0:
        return "minus " + _number_words(-number)
    if number < 20:
        return _ONES[number]
    if number < 100:
        rest = number % 10
        return _TENS[number // 10] + ("-" + _ONES[rest] if rest else "")
    for limit, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if number >= limit:
            head = _number_words(number // limit)
            rest = number % limit
            return f"{head} {name}" + (f" {_number_words(rest)}" if rest else "")
    return str(number)


def _cmd_choose(rest, sympy=None):
    numbers = [int(float(item)) for item in re.findall(r"\d+", str(rest or ""))]
    if len(numbers) < 2:
        raise MathError("Give me two whole numbers, like choose 52 5.")
    first, second = numbers[0], numbers[1]
    if re.search(r"\bperm", str(rest), re.IGNORECASE):
        if second > first:
            return "You cannot pick that many when there are not that many."
        return f"{first}P{second} = {math.perm(first, second)}"
    if second > first:
        return "You cannot pick that many when there are not that many."
    return f"{first}C{second} = {math.comb(first, second)}"


def _cmd_interest(rest, sympy=None):
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(rest or ""))]
    if len(numbers) < 3:
        raise MathError("Give me the amount, the rate and the years, like interest 1000 5 3")
    principal, rate, years = numbers[0], numbers[1], numbers[2]
    times = int(numbers[3]) if len(numbers) > 3 else 1
    if re.search(r"\bsimple\b", str(rest), re.IGNORECASE):
        amount = principal * (1 + rate / 100 * years)
        return f"simple interest: {principal:g} at {rate:g}% for {years:g} years = {_format_number(round(amount, 6))} (interest {_format_number(round(amount - principal, 6))})"
    amount = principal * (1 + rate / 100) ** years
    if re.search(r"\b(monthly|per month|compounded monthly)\b", str(rest), re.IGNORECASE):
        amount = principal * (1 + rate / 100 / 12) ** (years * 12)
    return (
        f"{principal:g} at {rate:g}% for {years:g} years, compounded {times} time(s) a year = "
        f"{_format_number(round(amount, 6))} (interest {_format_number(round(amount - principal, 6))})"
    )


def _cmd_percent(rest, sympy=None):
    text = str(rest or "").strip()
    numbers = [
        float(item) for item in
        re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text))
    ]
    if re.search(r"\b(change|difference|increase|decrease|growth)\b", text, re.IGNORECASE) and len(numbers) >= 2:
        old, new = numbers[0], numbers[1]
        if old == 0:
            raise MathError("A percentage change from zero has no meaning.")
        change = (new - old) / abs(old) * 100
        return f"from {old:g} to {new:g} is a change of {_format_number(round(change, 6))}%"
    if re.search(r"\bof\b", text, re.IGNORECASE) and len(numbers) >= 2:
        return f"{numbers[0]:g}% of {numbers[1]:g} = {_format_number(round(numbers[0] * numbers[1] / 100, 10))}"
    if len(numbers) >= 2:
        return f"{numbers[0]:g} is {_format_number(round(numbers[0] / numbers[1] * 100, 6))}% of {numbers[1]:g}"
    if numbers:
        return f"{numbers[0]:g}% = {_format_number(numbers[0] / 100)}"
    raise MathError("Give me the numbers, like percent 15 of 200.")


def _cmd_convert(rest, sympy=None):
    text = re.sub(
        r"^\s*(?:convert|how\s+many|what\s+is)\b", "",
        str(rest or "").strip(), flags=re.IGNORECASE,
    ).strip()
    match = re.match(r"(-?[\d.]+(?:[eE][-+]?\d+)?)\s*(\S+)\s+(?:in)?to\s+(\S+)\s*$", text, re.IGNORECASE)
    if not match:
        return "Give me it like this: convert 100 km to miles"
    amount, source, target = float(match.group(1)), match.group(2), match.group(3)
    return f"{amount:g} {source} = {_format_number(round(_convert_units(amount, source, target), 10))} {target}"


def _cmd_sum(rest, sympy=None):
    """Sums of arithmetic and geometric series, and sigma notation."""
    text = _clean_expression(rest)
    plain_range = re.search(r"(-?[\d.]+)\s*(?:to|through|upto|up to)\s*(-?[\d.]+)", text, re.IGNORECASE)
    if plain_range and not re.search(r"\barith|geo|nth|term|sigma", text, re.IGNORECASE):
        low, high = float(plain_range.group(1)), float(plain_range.group(2))
        if low != int(low) or high != int(high):
            raise MathError("Give me whole numbers for a sum like that.")
        low, high = int(low), int(high)
        if high < low:
            raise MathError("The end of the sum is before the start.")
        total = (high - low + 1) * (low + high) // 2
        return f"{low} to {high} adds up to {total}"
    sigma = re.search(r"sum\(([^,]+),\s*([A-Za-z_]\w*),\s*(-?\d+),\s*(-?\d+)\)", text, re.IGNORECASE)
    if sigma:
        body, low, high = sigma.group(1), int(sigma.group(3)), int(sigma.group(4))
        sympy_module = _lazy_sympy()
        variable = sympy_module.Symbol(sigma.group(2))
        value, _ = _evaluate_sympy(body)
        return f"sum from {low} to {high} = {_format_sympy(sympy_module.summation(value, (variable, low, high)))}"
    if re.search(r"\b(arith|arithmetic)\b", text, re.IGNORECASE):
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
        if len(numbers) >= 2:
            first, difference, count = numbers[0], numbers[1], int(numbers[2]) if len(numbers) > 2 else 10
            return (
                f"first {first:g}, difference {difference:g}, {count} terms: "
                f"sum = {_format_number(round(count * (2 * first + (count - 1) * difference) / 2, 10))}, "
                f"last term = {_format_number(first + (count - 1) * difference)}"
            )
    if re.search(r"\b(geo|geometric)\b", text, re.IGNORECASE):
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
        if len(numbers) >= 2:
            first, ratio, count = numbers[0], numbers[1], int(numbers[2]) if len(numbers) > 2 else 10
            if ratio == 1:
                total = first * count
            else:
                total = first * (ratio ** count - 1) / (ratio - 1)
            return (
                f"first {first:g}, ratio {ratio:g}, {count} terms: sum = "
                f"{_format_number(round(total, 10))}, last term = {_format_number(round(first * ratio ** (count - 1), 10))}"
            )
    if re.search(r"\bnth\b|\bterm\b", text, re.IGNORECASE):
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
        if len(numbers) >= 3:
            first, difference, n = numbers[0], numbers[1], int(numbers[2])
            return f"term {n} = {_format_number(first + (n - 1) * difference)}"
    value, _ = _evaluate_sympy(text)
    return _format_sympy(value)


def _cmd_nth_prime(rest, sympy=None):
    numbers = [int(float(item)) for item in re.findall(r"\d+", str(rest or ""))]
    sympy_module = _lazy_sympy()
    if not numbers:
        raise MathError("Tell me which prime, like nth prime 47")
    which = numbers[0]
    if which > 10 ** 7:
        raise MathError("That is past the end of what I can count to.")
    return f"the {which}{'th' if which not in (11, 12, 13) else 'th'} prime number is {sympy_module.prime(which)}"


def _cmd_prime_list(rest, sympy=None):
    numbers = [int(float(item)) for item in re.findall(r"\d+", str(rest or ""))]
    sympy_module = _lazy_sympy()
    if not numbers:
        raise MathError("Tell me how many primes, like primes 20")
    count = numbers[0]
    if count > 2000:
        raise MathError("That is more primes than I will print.")
    primes = list(sympy_module.primerange(2, 10 ** 6))[:count]
    return f"the first {count} primes: " + ", ".join(str(item) for item in primes)


def _cmd_truth_table(rest, sympy=None):
    match = re.match(r"(.+?)\s+(?:is|as|for)\s+(.+)(\s*and\s*.+)?$", str(rest or "").strip(), re.IGNORECASE)
    if not match:
        return "Give me it like this: truth table (p and q) is false"
    sympy_module = _lazy_sympy()
    expression = _clean_expression(match.group(1))
    false_value = match.group(2).strip().lower() in ("false", "0", "no")
    symbols = sorted(sympy_module.symbols(expression), key=str)
    if len(symbols) > 4:
        raise MathError("A truth table of that many variables is too wide to read.")
    rows = []
    for values in _product([(False, True)] * len(symbols)):
        substitutions = dict(zip(symbols, values))
        result = bool(sympy_module.simplify(expression.subs(substitutions)))
        rows.append(
            " | ".join(f"{name}={'T' if value else 'F'}" for name, value in substitutions.items())
            + f" | {expression} = {'T' if result else 'F'}"
        )
    header = " | ".join(str(name) for name in symbols) + f" | {expression}"
    return header + "\n" + "\n".join(rows) + (f"\n(treated as false: {match.group(2).strip()})" if false_value else "")


def _product(pools):
    result = [()]
    for pool in pools:
        result = [existing + (item,) for existing in result for item in pool]
    return result


def _cmd_units(rest, sympy=None):
    return (
        "convert 100 km to miles | convert 98.6 f to c | convert 5 kg to lb | "
        "convert 1.5 gb to mb | convert 90 min to hours"
    )


def _cmd_help(rest, sympy=None):
    return (
        "Just give me the maths: 123456789 * 987654321, or 2**100, or sqrt(2), or "
        "0.1+0.2. For anything else use a word first: solve x**2-5x+6=0 | "
        "diff x**3 wrt x | integrate x**2 dx | integrate x**2 from 0 to 3 | "
        "simplify (x**2-1)/(x-1) | factor x**2-5x+6 | expand (x+1)**8 | "
        "limit sin(x)/x as x -> 0 | series exp(x) around x = 0 to 6 | "
        "substitute x**2+1 with x = 3 | matrix [[1,2],[3,4]] det | "
        "mean 3 7 11 | median 4 1 9 2 | stdev 2 4 4 4 5 5 7 9 | "
        "prime 104729 | factor 360 | divisors 360 | gcd 84 126 | lcm 4 6 | "
        "choose 52 5 | perm 10 3 | factorial 20 | fibonacci 30 | "
        "interest 1000 5 3 | percent change 80 to 100 | percent 15 of 200 | "
        "convert 100 km to miles | base 255 to hex | numeric sqrt(2) | "
        "truth table (p and q) or (p or q) | primes 10"
    )


COMMANDS = {
    "solve": _cmd_solve, "roots": _cmd_solve,
    "diff": _cmd_diff, "differentiate": _cmd_diff, "derivative": _cmd_diff,
    "integrate": _cmd_integrate, "integral": _cmd_integrate, "antiderivative": _cmd_integrate,
    "simplify": _cmd_simplify, "expand": _cmd_expand, "factor": _cmd_factor,
    "limit": _cmd_limit, "series": _cmd_series, "expand_around": _cmd_series,
    "substitute": _cmd_substitute, "sub": _cmd_substitute, "plug": _cmd_substitute,
    "matrix": _cmd_matrix, "determinant": _cmd_matrix, "det": _cmd_matrix,
    "inverse": _cmd_matrix, "eigenvalues": _cmd_matrix,
    "numeric": _cmd_numeric, "approx": _cmd_numeric, "decimal": _cmd_numeric,
    "mean": _cmd_statistics, "average": _cmd_statistics, "median": _cmd_statistics,
    "mode": _cmd_statistics, "stdev": _cmd_statistics, "std": _cmd_statistics,
    "variance": _cmd_statistics, "stats": _cmd_statistics, "statistics": _cmd_statistics,
    "range": _cmd_statistics, "sum": _cmd_sum, "series_sum": _cmd_sum,
    "prime": _cmd_numbers, "factorint": _cmd_numbers, "divisors": _cmd_numbers,
    "nthprime": _cmd_nth_prime, "nth": _cmd_nth_prime,
    "gcd": _cmd_numbers, "lcm": _cmd_numbers, "factorial": _cmd_numbers,
    "fibonacci": _cmd_numbers, "sqrt": _cmd_numbers, "base": _cmd_numbers,
    "words": _cmd_numbers, "primes": _cmd_prime_list,
    "choose": _cmd_choose, "binomial": _cmd_choose, "comb": _cmd_choose, "perm": _cmd_choose,
    "interest": _cmd_interest, "compound": _cmd_interest,
    "percent": _cmd_percent, "percentage": _cmd_percent,
    "convert": _cmd_convert, "units": _cmd_units,
    "truth": _cmd_truth_table, "help": _cmd_help, "examples": _cmd_help,
}
# A command that needs sympy, so "help" and the number-theoretic shortcuts work
# even where sympy itself is not importable.
_SYMPY_COMMANDS = {
    "solve", "roots", "diff", "differentiate", "derivative", "integrate",
    "integral", "antiderivative", "simplify", "expand", "factor", "limit",
    "series", "expand_around", "substitute", "sub", "plug", "matrix",
    "determinant", "det", "inverse", "eigenvalues", "numeric", "approx",
    "decimal", "prime", "factorint", "divisors", "primes", "truth", "sum",
    "series_sum",
}
# Commands that are read as a whole, verb included.
_VERB_COMMANDS = {
    "prime", "factorint", "divisors", "gcd", "lcm", "factorial", "fibonacci",
    "nthprime", "nth",
    "sqrt", "base", "words", "mean", "average", "median", "mode", "stdev",
    "std", "variance", "stats", "statistics", "range", "sum", "series_sum",
    "choose", "binomial", "comb", "perm", "interest", "compound", "percent",
    "percentage", "convert", "units", "matrix", "determinant", "det", "inverse",
    "eigenvalues", "primes", "truth", "help", "examples",
}
_COMMAND_WORDS = "|".join(sorted(COMMANDS, key=len, reverse=True))


def calculate(argument):
    """Answer a maths question, exactly. See the module docstring for why.

    Returns a line meant to be read out or dropped straight into her reply. It
    never raises: a question she cannot do comes back as a sentence she can say,
    because the alternative is a traceback in the channel.
    """
    try:
        text = _check_text(argument)
    except MathError as error:
        # A refusal is an answer she can say, not an exception: this runs inside
        # her tool loop, where an exception takes the whole turn down.
        return str(error)
    if not text:
        return "There was nothing to work out."
    try:
        # A command is a word then some maths. "sum 1 to 100" and "prime 97" both
        # arrive here, and a bare expression such as "2 + 2" does not.
        match = re.match(rf"^({_COMMAND_WORDS})\b[:\s]+(.+)$", text, re.IGNORECASE | re.DOTALL)
        bare_command = re.fullmatch(rf"({_COMMAND_WORDS})", text, re.IGNORECASE)
        if bare_command:
            # "solve" on its own is not a question, and answering it with the
            # word "solve" looks like a working one.
            return "Tell me what to solve, like solve x**2 - 5x + 6 = 0"
        if match:
            name = match.group(1).lower()
            rest = match.group(2).strip()
            handler = COMMANDS[name]
            sympy_module = _lazy_sympy() if name in _SYMPY_COMMANDS else None
            # The verb is part of the question for the commands that decide what
            # to do by reading it: handed only "20", "factorial 20" cannot tell a
            # factorial from a birthday, and answered "twenty".
            payload = f"{name} {rest}" if name in _VERB_COMMANDS else rest
            result = handler(payload, sympy_module)
        elif re.match(
            r"^\s*-?[\d.]+(?:[eE][-+]?\d+)?\s+\S+\s+(?:in)?to\s+\S+\s*$", text, re.IGNORECASE
        ) and not re.search(r"[+\-*/^]", text):
            # "100 km to miles" with no verb on the front. A model forgets the
            # word more often than it forgets the units, and this is the shape it
            # writes when it does.
            result = _cmd_convert(text, None)
        elif "=" in text and not text.startswith("="):
            # An equation with no verb is a solve with no verb. "2x + 3 = 11" is
            # not a parsable expression, and it is how the question gets asked.
            result = _cmd_solve(text, _lazy_sympy())
        else:
            result = _evaluate_direct(text)
        result = str(result).strip()
        if not result:
            raise MathError("I worked it out and got nothing back.")
        return result[:MAX_OUTPUT_CHARS]
    except MathError as error:
        return str(error)
    except RecursionError:
        return "That is too deeply nested for me to work out."
    except MemoryError:
        return "That is too big for me to work out."
    except Exception as error:  # anything sympy throws that is not one of ours
        detail = str(error).split("\n")[0][:160]
        return f"I could not work that one out: {detail}"


def _evaluate_direct(text):
    """A bare expression, exactly when sympy can and safely, and in Python when
    it cannot.

    The Python layer is the fallback, not the decoration: it is what answers when
    sympy is missing, and it is what answers a plain arithmetic question with no
    parsing surprises at all.
    """
    cleaned = _clean_expression(text)
    if not cleaned:
        raise MathError("There was nothing to work out.")
    # Prose parses as a product of words once implicit multiplication is on, so
    # "how do I bake bread" came back as I*bake*bread*do*how. Words with no digit
    # anywhere are not a calculation, and saying so is more useful than a
    # multiplication of sentences.
    if not re.search(r"\d", cleaned):
        names = [
            name for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", cleaned)
            if name not in _FUNCTIONS and name not in _CONSTANTS
        ]
        if len(names) > 2 or (len(names) == 1 and len(cleaned) > 24):
            raise MathError("That does not look like a maths question to me.")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        tree = None
    if tree is not None and _is_plain_arithmetic(tree):
        # Numbers and the four operations only. This is the path that keeps
        # working when sympy is not installed, and it is exact for whole numbers,
        # which is what "123456789 * 987654321" needs.
        return _format_number(_eval_node(tree, _VARIABLES))
    try:
        value, _ = _evaluate_sympy(cleaned)
        if sympy_available(value):
            # (2+3i)**8 left unexpanded is a restatement of the question. The
            # answer is the 6561 - 432i, which is the point of asking.
            # power_base is the part that matters: expand() leaves (2+3*I)**8
            # folded up unless it is told to take the sum apart first, which
            # would answer a question with a restatement of it.
            return _format_sympy(
                _lazy_sympy().expand(
                    value, mul=True, power_exp=True, power_base=True, multinomial=True,
                )
            )
        return _format_sympy(value)
    except MathError:
        if tree is None:
            raise
        # A function she has in Python but sympy has not heard of, or a name with
        # no value: evaluate it in Python rather than refusing.
        variables = dict(_VARIABLES)
        for name in _symbols_in(cleaned):
            variables.setdefault(name, 0)
        return _format_number(_eval_node(tree, variables))


def sympy_available(value):
    """Does this result contain the imaginary unit, so expanding it says more?

    Asked of the parsed result rather than the text on purpose. Looking for a
    lone "i" in the string never matches anything, because there is no word
    boundary between a digit and the letter next to it - "3i" is one word to a
    regular expression - and the expansion silently did not happen.
    """
    try:
        return value.has(_lazy_sympy().I)
    except (AttributeError, TypeError, ValueError):
        return False


def _is_plain_arithmetic(tree):
    """Is this only numbers, brackets and + - * / // % ** ? Nothing symbolic."""
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            continue
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            continue
        if isinstance(node, (ast.operator, ast.unaryop)):
            # The + itself, walked on its own account. Missing this made every
            # plain calculation look symbolic, so 0.1+0.2 came back from sympy
            # as 0.300000000000000 instead of 0.3.
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return False
            continue
        if isinstance(node, ast.Expression):
            continue
        return False
    return True



# A question asked in a sentence rather than in maths notation. This is the layer
# that matters in a chat, where nobody types "calculate 1234*5678" - they type
# "what is 1234 times 5678" and expect a number back. Each pattern rewrites the
# sentence into the command that answers it.
_REQUEST_PATTERNS = (
    # "the first 5 primes" - the filler rules strip "the" and leave the words
    # in the wrong order, so this has to be rewritten whole.
    (r"\bthe\s+first\s+(\d+)\s+primes?\b", r"primes \1"),
    # "how many days in 3 years" is a conversion of a quantity, not of one unit
    (r"\bhow\s+many\s+([a-z]+)\s+in\s+(-?[\d.,]+)\s+([a-z]+)", r"convert \2 \3 to \1"),
    # "the 47th prime", "what is the 100th prime number"
    (r"\b(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+prime(?:\s+number)?\b", r"nth prime \1"),
    # "the square root of 144", "square root of 2"
    (r"\bsquare\s+root\s+of\s+(-?[\d.]+)", r"sqrt \1"),
    (r"\bcube\s+root\s+of\s+(-?[\d.]+)", r"(\1)**(1/3)"),
    # "what is 15 percent of 240", "15% of 240"
    (r"(-?[\d.,]+)\s*(?:%|percent(?:age)?)\s+of\s+(-?[\d.,]+)", r"percent \1 of \2"),
    # "how much is 15 percent of 1847.63"
    (r"percent(?:age)?\s+of\s+(-?[\d.,]+)\s+is\s+(-?[\d.,]+)", r"percent \1 of \2"),
    # "convert 100 km into miles", "how many km is 100 miles"
    (r"\bhow\s+many\s+([a-z]+)\s+(?:is|are)\s+(-?[\d.]+)\s+([a-z]+)", r"convert \2 \3 to \1"),
    (r"\bhow\s+many\s+([a-z]+)\s+(?:in|per)\s+(?:a|one|an)\s+([a-z]+)", r"convert 1 \2 to \1"),
    # "what is 45 factorial", "factorial of 20"
    (r"\b(\d+)\s*(?:!|factorial)\b", r"factorial \1"),
    (r"\bfactorial\s+of\s+(\d+)", r"factorial \1"),
    # "what is 10 choose 3", "10 permutations of 3"
    (r"\b(\d+)\s+choose\s+(\d+)", r"choose \1 \2"),
    (r"\b(\d+)\s+permutations?\s+of\s+(\d+)", r"perm \1 \2"),
    # "divide 100 by 7"
    (r"\bdivide\s+(-?[\d.,]+)\s+by\s+(-?[\d.,]+)", r"(\1)/(\2)"),
    # "x to the power of 5", "2 squared", "3 cubed"
    (r"(-?[\d.,]+)\s*(?:to the power of|raised to(?: the power of)?)\s*(-?[\d.,]+)", r"(\1)**(\2)"),
    (r"\b(-?[\d.,]+)\s+squared\b", r"(\1)**2"),
    (r"\b(-?[\d.,]+)\s+cubed\b", r"(\1)**3"),
    # "add 5 and 7", "multiply 6 by 8", "subtract 3 from 10", "divide 100 by 4"
    (r"\badd\s+(-?[\d.,]+)\s+(?:and|to)\s+(-?[\d.,]+)", r"(\1)+(\2)"),
    (r"\bmultiply\s+(-?[\d.,]+)\s+by\s+(-?[\d.,]+)", r"(\1)*(\2)"),
    (r"\bsubtract\s+(-?[\d.,]+)\s+from\s+(-?[\d.,]+)", r"(\2)-(\1)"),
    # "average of 3, 7 and 11" -> the statistics command reads the numbers anyway
    (r"\b(average|mean|median|mode|standard deviation|stdev|variance|sum|range)\s+of\s+(.+)",
     r"\1 \2"),
    # "difference between 30 and 12"
    (r"\bdifference\s+between\s+(-?[\d.,]+)\s+and\s+(-?[\d.,]+)", r"(\1)-(\2)"),
    # "the derivative of x^3" / "integrate x^2"
    (r"\bderivative\s+of\s+(.+)", r"diff \1"),
    (r"\bintegral\s+of\s+(.+)", r"integrate \1"),
    (r"\bsquare\s+root\b", "sqrt"),
    (r"\btimes\b", "*"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bover\b", "/"),
    (r"\bdivided by\b", "/"),
    (r"\bmultiplied by\b", "*"),
)
# What a person says round the actual question.
_REQUEST_FILLER = re.compile(
    r"^\s*(?:"
    r"hey|kookie|cookie|please|can you|could you|tell me|do you know|i wonder|"
    r"i want to know|can you tell me|work out|figure out|"
    r"what(?:'s| is| are| was| were)?|whats|how much is|how much are|how many is|"
    r"how many are|how many|how do i|how do you|"
    r"the|a|an|of|me|us|to|for|that|this|it|and|is|are|was|were|be|"
    r"calculate|compute|evaluate|give me|tell me|about|"
    r"quickly|right now|now|exactly|precisely|again"
    r")\b[\s,]*",
    re.IGNORECASE,
)


def calculate_request(text):
    """Answer a maths question however it was actually phrased.

    The whole point of the tool is that nobody types maths into a chat window, so
    this is the entry point that matters: "what is 15 percent of 1,847.63" and
    "987654321987 * 123456789123" both have to come back with a number. Returns
    "" when the text is not maths at all, which is how the caller knows to leave
    the message to the model.
    """
    raw = str(text or "").strip()
    if not raw or len(raw) > 400:
        return ""
    working = re.sub(r"[?!.]+$", "", raw).strip()
    # Numbers first: the phrasing rules are mostly about what sits between them.
    for pattern, replacement in _REQUEST_PATTERNS:
        if re.search(pattern, working, re.IGNORECASE):
            working = re.sub(pattern, replacement, working, flags=re.IGNORECASE)
            break
    previous = None
    while previous != working:
        previous = working
        working = _REQUEST_FILLER.sub("", working).strip()
    working = re.sub(r"\s+", " ", working).strip()
    if not working or len(working) < 2:
        return ""
    if not re.search(r"[\d]", working) and not re.match(
        rf"^({_COMMAND_WORDS})\b", working, re.IGNORECASE
    ):
        return ""
    answer = calculate(working)
    if not answer or answer.lower().startswith("that does not look like a maths"):
        return ""
    return answer


def preload():
    """Load sympy now rather than on the first question.

    Importing it costs about half a second and tens of megabytes, which is
    nothing next to the local model and a lot next to a one-line question nobody
    has asked yet. Called from the background at startup so the first person who
    asks her to multiply two big numbers does not pay for it.
    """
    try:
        _lazy_sympy()
    except MathError:
        return False
    return True


def tool_description():
    """The schema text, kept next to the code so the two cannot drift."""
    return (
        "Do any maths exactly, instead of working it out in your head and getting "
        "it wrong. Pass the maths as it is written: an expression like "
        "'123456789 * 987654321', '2**100', 'sqrt(2)', '(2+3i)**8', or a word "
        "first for anything else: solve, diff, integrate, simplify, expand, "
        "factor, limit, series, substitute, matrix, mean, median, stdev, prime, "
        "divisors, gcd, lcm, choose, factorial, fibonacci, interest, percent, "
        "convert, base, numeric, truth, primes. Call 'calculate examples' for "
        "the full list. Use this for any number she is asked for - she cannot be "
        "trusted to do arithmetic, and a wrong number said confidently is worse "
        "than no number."
    )
