"""Code structure becomes page structure: headers, parameter tables, branch layouts."""
from __future__ import annotations

import re

from python_beautifier import beautify

from conftest import classes_in, visible_text


def table_rows(html: str, card_id: str):
    """Header cells and body rows of one card's parameters table (names are the ``<code>`` text)."""
    card = html[html.index(f'id="{card_id}"'):]
    table = card[card.index("<table"):card.index("</table>")]
    head = [re.sub(r"<[^>]+>", "", h) for h in re.findall(r"<th[^>]*>(.*?)</th>", table)]
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S)
        if not cells:
            continue
        name = re.search(r"<code>(.*?)</code>", cells[0]).group(1)
        rows.append([name] + [re.sub(r"<[^>]+>", "", c).strip() for c in cells[1:]])
    return head, rows


DOCUMENTED = '''
def connect(host: str, port: int = 8080, *, retries=3, **options):
    """Open a connection.

    Args:
        host: Name of the machine.
        port (int): TCP port to use.
        retries: How often to try again.
        **options: Passed on to the socket.
    """
    return host, port, retries, options
'''


def test_function_card_and_parameter_table():
    html = beautify(DOCUMENTED)
    assert 'id="d-connect"' in html
    head, rows = table_rows(html, "d-connect")
    assert head == ["Name", "Type", "Default", "Comments"]
    by_name = {r[0].lstrip("*"): r for r in rows}
    assert set(by_name) == {"host", "port", "retries", "options"}
    assert by_name["host"][1] == "str" and "Name of the machine" in by_name["host"][3]
    assert by_name["port"][1] == "int" and "8080" in by_name["port"][2]
    assert "How often to try again" in by_name["retries"][3]
    assert "Passed on to the socket" in by_name["options"][3]
    assert by_name["options"][0] == "**options"


def test_types_come_from_annotations_then_docstrings_then_defaults():
    code = '''
def f(a: float, b, c=3):
    """Doc.

    Args:
        a (int): ignored, the annotation wins.
        b (str): taken from the docstring.
    """
'''
    _, rows = table_rows(beautify(code), "d-f")
    types = {r[0]: r[1] for r in rows}
    assert types["a"].startswith("float")
    assert types["b"].startswith("str")
    assert types["c"].startswith("int")  # inferred from the default


def test_comments_fall_back_to_inline_source_comments():
    code = "def f(\n    a,  # the first one\n    b,\n):\n    return a, b\n"
    _, rows = table_rows(beautify(code), "d-f")
    assert "the first one" in rows[0][3]


def test_self_and_cls_are_not_listed():
    code = "class A:\n    def m(self, x): ...\n    @classmethod\n    def c(cls, y): ...\n"
    html = beautify(code)
    _, rows = table_rows(html, "d-A.m")
    assert [r[0] for r in rows] == ["x"]
    _, rows = table_rows(html, "d-A.c")
    assert [r[0] for r in rows] == ["y"]


def test_classes_methods_and_nesting_get_their_own_cards():
    code = "class Outer:\n    class Inner:\n        def deep(self): ...\n    def meth(self): ...\n\ndef free(): ...\n"
    html = beautify(code)
    for ident in ("d-Outer", "d-Outer.Inner", "d-Outer.Inner.deep", "d-Outer.meth", "d-free"):
        assert f'id="{ident}"' in html


def test_short_if_else_is_drawn_as_two_columns_true_left_false_right():
    html = beautify("def f(x):\n    if x:\n        a = 1\n    else:\n        a = 2\n    return a\n")
    assert "split" in classes_in(html)
    left, right = html.index('class="col yes'), html.index('class="col no')
    assert left < right
    assert "a = 1" in visible_text(html[left:right]) and "a = 2" in visible_text(html[right:])


def test_elif_chains_become_a_ladder():
    code = "def f(x):\n    if x == 1:\n        a = 1\n    elif x == 2:\n        a = 2\n    else:\n        a = 3\n    return a\n"
    assert "ladder" in classes_in(beautify(code))


def test_early_exit_conditions_become_guards():
    code = 'def f(x):\n    if x is None:\n        raise ValueError("x")\n    return x\n'
    assert "guard" in classes_in(beautify(code))


def test_try_except_is_laid_out_in_lanes():
    code = "def load(p):\n    try:\n        d = open(p).read()\n    except OSError:\n        return None\n    return d\n"
    classes = classes_in(beautify(code))
    assert {"lanes", "lane"} <= classes


def test_long_branches_are_stacked_instead_of_squeezed():
    body = lambda prefix: "".join(  # noqa: E731
        f'        {prefix}{i} = compute_something_long_name(x, {i}, "some literal string number {i}")\n' for i in range(9)
    )
    code = "def g(x):\n    if x:\n" + body("a") + "    else:\n" + body("b") + "    return 1\n"
    classes = classes_in(beautify(code))
    assert "stack" in classes and "split" not in classes


def test_loops_with_and_match_have_their_own_blocks():
    code = "def f(xs):\n    for x in xs:\n        print(x)\n    with open('f') as fh:\n        fh.read()\n    match xs:\n        case []:\n            return 0\n        case _:\n            return 1\n"
    assert {"b-loop", "b-with", "b-match"} <= classes_in(beautify(code))


def test_banner_todo_and_plain_comments_are_recognised():
    code = "# ---- Section ----\nX = 1  # trailing\n# TODO: fix this\ndef h(x):\n    # note above\n    return x\n"
    html = beautify(code)
    assert {"rule", "todo", "note"} <= classes_in(html)
    assert "fix this" in visible_text(html)


def test_complexity_is_reported():
    simple = beautify("def f(): return 1\n")
    branchy = beautify("def f(x):\n" + "".join(f"    if x == {i}:\n        x += 1\n" for i in range(12)))
    assert "complexity 1 (A)" in simple
    assert re.search(r"complexity 1[3-9] \(C\)", branchy)


def test_decorators_async_and_generators_are_flagged():
    code = "import functools\n@functools.lru_cache\nasync def a(): ...\ndef g():\n    yield 1\n"
    html = beautify(code)
    text = visible_text(html).lower()
    assert "async" in text and "generator" in text


def decorator_chips(html: str, card_id: str):
    """The ``@decorator`` chips of one card's header."""
    card = html[html.index(f'id="{card_id}"'):]
    header = card[:card.index("</header>")]
    return [re.sub(r"<[^>]+>", "", c) for c in re.findall(r'<code class="deco"[^>]*>(.*?)</code>', header, flags=re.S)]


def test_decorators_are_shown_once_and_never_dropped():
    code = (
        "from typing import final\n"
        "class A:\n"
        "    @staticmethod\n"
        "    def a(): ...\n"
        "    @final\n"
        "    def b(self): ...\n"
        "    @some.custom(1)\n"
        "    @property\n"
        "    def c(self): ...\n"
    )
    html = beautify(code)
    assert decorator_chips(html, "d-A.a") == []  # already shown as the "static" badge
    assert decorator_chips(html, "d-A.b") == ["@final"]  # no badge for it, so the chip must stay
    assert decorator_chips(html, "d-A.c") == ["@some.custom(1)"]  # "property" is a badge, the rest stays


def test_module_overview_lists_every_definition():
    code = "class A:\n    def m(self): ...\ndef f(): ...\ndef g(): ...\n"
    html = beautify(code)
    for name in ("A", "A.m", "f", "g"):
        assert f'href="#d-{name}"' in html
