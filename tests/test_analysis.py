"""Complexity, flow-ending and type-inference helpers."""
from __future__ import annotations

import ast
import textwrap

import pytest

from python_beautifier import analysis


def func(code: str) -> ast.AST:
    return ast.parse(textwrap.dedent(code)).body[0]


@pytest.mark.parametrize(
    "body, expected",
    [
        ("return 1", 1),
        ("if a: pass", 2),
        ("if a: pass\nelif b: pass\nelse: pass", 3),
        ("for x in y: pass", 2),
        ("while a: pass", 2),
        ("x = a and b and c", 3),
        ("x = 1 if a else 2", 2),
        ("x = [i for i in y if i if j]", 4),
        ("try:\n    pass\nexcept A:\n    pass\nexcept B:\n    pass", 3),
        ("assert a", 2),
        ("match a:\n    case 1: pass\n    case 2: pass\n    case _: pass", 3),
    ],
)
def test_cyclomatic_complexity(body, expected):
    code = "def f(a, b, c, y, j):\n" + textwrap.indent(body, "    ") + "\n"
    assert analysis.complexity(func(code)) == expected


def test_nested_functions_do_not_count_towards_the_outer_function():
    code = "def outer():\n    def inner(x):\n        if x:\n            return 1\n    return inner\n"
    assert analysis.complexity(func(code)) == 1


@pytest.mark.parametrize("cc, letter", [(1, "A"), (5, "A"), (6, "B"), (10, "B"), (11, "C"), (20, "C"), (21, "D"), (31, "E"), (99, "F")])
def test_grades(cc, letter):
    assert analysis.grade(cc) == letter


@pytest.mark.parametrize(
    "code, expected",
    [
        ("return 1", "return"),
        ("raise ValueError", "raise"),
        ("pass", None),
        ("if a:\n    return 1\nelse:\n    raise E", "return"),
        ("if a:\n    return 1", None),
        ("try:\n    return 1\nexcept E:\n    raise", "return"),
        ("try:\n    x = 1\nexcept E:\n    raise", None),
        ("with a:\n    return 1", "return"),
        ("sys.exit(1)", "exit"),
    ],
)
def test_ends_flow(code, expected):
    stmts = ast.parse(code).body
    assert analysis.ends_flow(stmts) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [("1", "int"), ("'s'", "str"), ("None", "None"), ("[]", "list"), ("{}", "dict"), ("(1,)", "tuple"), ("{1}", "set"), ("-1.5", "float"), ("f'{x}'", "str"), ("lambda: 0", "Callable"), ("Foo()", "Foo")],
)
def test_infer_type(expr, expected):
    assert analysis.infer_type(ast.parse(expr, mode="eval").body) == expected


def test_generators_and_awaits_are_detected():
    assert analysis.is_generator(func("def f():\n    yield 1"))
    assert not analysis.is_generator(func("def f():\n    return [(yield)]\n".replace("[(yield)]", "1")))
    assert analysis.has_await(func("async def f():\n    await g()"))
    assert not analysis.has_await(func("async def f():\n    return 1"))
