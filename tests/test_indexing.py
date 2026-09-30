"""NumPy / PyTorch style subscripts such as ``img[:, n:, ...]`` are decoded axis by axis."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Tuple

import pytest

from python_beautifier import beautify
from python_beautifier.highlight import render_items
from python_beautifier.indexing import MINUS, ordinal, plain
from python_beautifier.source import Source

from conftest import ROOT, render_checked

EXAMPLE = ROOT / "examples" / "tensor_indexing.py"

ALL = ("all", "axis 0", "all")
OTHERS = ("others", "others", "further axes stay whole")


def cells(code: str) -> List[Tuple[str, str, str]]:
    """(kind, label, sentence) of every strip cell for the first decoded subscript of *code*.

    The code is placed in a file that imports NumPy, where even a bare ``None`` index counts.
    """
    guides = Source("import numpy as np\n" + code).index.guides
    assert guides, f"nothing was decoded in {code!r}"
    return [(c.kind, c.label, plain(c.meaning)) for c in guides[0].cells]


# --------------------------------------------------------------------------- #
#  what the terms mean
# --------------------------------------------------------------------------- #


def test_the_motivating_example():
    assert cells("img = img[:, num_txt_tokens:, ...]") == [
        ("all", "axis 0", "all"),
        ("slice", "axis 1", "from `num_txt_tokens` to the end"),
        ("rest", "rest", "all remaining axes"),
    ]


MEANINGS = {
    # picks, and the axes that nothing mentions
    "x[:, 0]": [ALL, ("pick", "axis 1", "pick item 0 \u00b7 axis removed"), OTHERS],
    "x[:, -1]": [ALL, ("pick", "axis 1", "pick the last item \u00b7 axis removed"), OTHERS],
    "x[:, -2]": [ALL, ("pick", "axis 1", "pick item -2 (2nd from the end) \u00b7 axis removed"), OTHERS],
    "x[:, i]": [ALL, ("pick", "axis 1", "index with `i`"), OTHERS],
    # ellipsis: axes after it count from the end
    "x[..., 0]": [("rest", "rest", "all leading axes"), ("pick", f"axis {MINUS}1", "pick item 0 \u00b7 axis removed")],
    "x[0, ..., 1]": [
        ("pick", "axis 0", "pick item 0 \u00b7 axis removed"),
        ("rest", "rest", "all axes in between"),
        ("pick", f"axis {MINUS}1", "pick item 1 \u00b7 axis removed"),
    ],
    "x[...]": [("rest", "rest", "every axis (the whole array)")],
    "x[..., :, 0]": [
        ("rest", "rest", "all leading axes"),
        ("all", f"axis {MINUS}2", "all"),
        ("pick", f"axis {MINUS}1", "pick item 0 \u00b7 axis removed"),
    ],
    # None adds an axis but does not use one up
    "x[:, None, :]": [ALL, ("new", "new axis", "new axis of length 1"), ("all", "axis 1", "all"), OTHERS],
    "x[None]": [("new", "new axis", "new axis of length 1"), OTHERS],
    "x[..., None]": [("rest", "rest", "all leading axes"), ("new", "new axis", "new axis of length 1")],
    "x[:, np.newaxis]": [ALL, ("new", "new axis", "new axis of length 1"), OTHERS],
    # slices
    "x[::2, ::-1]": [("slice", "axis 0", "every 2nd item"), ("slice", "axis 1", "reversed"), OTHERS],
    "x[1:, :-1]": [("slice", "axis 0", "skip the first item"), ("slice", "axis 1", "all but the last item"), OTHERS],
    "x[-3:, :5]": [("slice", "axis 0", "the last 3"), ("slice", "axis 1", "the first 5"), OTHERS],
    "x[-1:, :1]": [
        ("slice", "axis 0", "only the last item \u00b7 axis kept"),
        ("slice", "axis 1", "only the first item \u00b7 axis kept"),
        OTHERS,
    ],
    "x[:, i:i + 1]": [ALL, ("slice", "axis 1", "only item `i` \u00b7 axis kept"), OTHERS],
    "x[:, i:i + 2]": [ALL, ("slice", "axis 1", "from `i` up to `i + 2`, not included"), OTHERS],
    "x[:-3, 3:]": [("slice", "axis 0", "all but the last 3"), ("slice", "axis 1", "skip the first 3"), OTHERS],
    "x[2:7, 1:2]": [("slice", "axis 0", "items 2 to 6"), ("slice", "axis 1", "only item 1 \u00b7 axis kept"), OTHERS],
    "x[1:-1, 0:4]": [("slice", "axis 0", "skip the first 1 and the last item"), ("slice", "axis 1", "the first 4"), OTHERS],
    "x[1::2, 0:10:3]": [
        ("slice", "axis 0", "skip the first item \u00b7 every 2nd item"),
        ("slice", "axis 1", "the first 10 \u00b7 every 3rd item"),
        OTHERS,
    ],
    "x[a:b, c:]": [
        ("slice", "axis 0", "from `a` up to `b`, not included"),
        ("slice", "axis 1", "from `c` to the end"),
        OTHERS,
    ],
    "x[:n, ::s]": [
        ("slice", "axis 0", "up to `n`, not included"),
        ("slice", "axis 1", "every `s`-th item"),
        OTHERS,
    ],
    "x[5:1:-1, :]": [("slice", "axis 0", "start `5`, stop `1`, step `-1`"), ("all", "axis 1", "all"), OTHERS],
    # masks, index lists
    "x[mask, :]": [("pick", "axis 0", "index with `mask`"), ("all", "axis 1", "all"), OTHERS],
    "x[x > 0, :]": [("mask", "axis 0", "keep where `x > 0`"), ("all", "axis 1", "all"), OTHERS],
    "x[~m, ...]": [("mask", "axis 0", "keep where `~m`"), ("rest", "rest", "all remaining axes")],
    "x[[0, 2], :]": [("take", "axis 0", "gather positions `[0, 2]`"), ("all", "axis 1", "all"), OTHERS],
    "x[slice(1, 3), :]": [("slice", "axis 0", "slice object `slice(1, 3)`"), ("all", "axis 1", "all"), OTHERS],
    # long runs of trivial terms share one cell
    "x[:, :, :, 0]": [
        ("all", "axes 0\u20132", "all"),
        ("pick", "axis 3", "pick item 0 \u00b7 axis removed"),
        OTHERS,
    ],
    "m[None, None, :, :]": [("new", "new axes", "2 new axes of length 1"), ALL, ("all", "axis 1", "all"), OTHERS],
    "x[..., :, :, :]": [("rest", "rest", "all leading axes"), ("all", f"axes {MINUS}3 to {MINUS}1", "all")],
    # pandas: rows and columns; .loc slices are by label and include their end
    "df.iloc[:5, 1:]": [("slice", "rows", "the first 5"), ("slice", "columns", "skip the first item")],
    "df.loc[:, cols]": [("all", "rows", "all"), ("pick", "columns", "label(s) `cols`")],
    'df.loc[:, "a":"c"]': [
        ("all", "rows", "all"),
        ("slice", "columns", 'from `"a"` through `"c"` (both included)'),
    ],
    "df.loc[lo:hi, :]": [("slice", "rows", "from `lo` through `hi` (both included)"), ("all", "columns", "all")],
    "df.loc[:end, :]": [("slice", "rows", "up to and including `end`"), ("all", "columns", "all")],
    "df.loc[['a', 'b'], :]": [("take", "rows", "select labels `['a', 'b']`"), ("all", "columns", "all")],
    "df.iloc[[0, 2], :]": [("take", "rows", "gather positions `[0, 2]`"), ("all", "columns", "all")],
    "df.loc[(slice(None), 1), :]": [("pick", "rows", "MultiIndex key `(slice(None), 1)`"), ("all", "columns", "all")],
    "df.loc(axis=0)[:, 'x']": [("all", "rows", "all"), ("pick", "columns", 'label `"x"`'.replace('"', "'"))],
}


@pytest.mark.parametrize("code", list(MEANINGS))
def test_what_each_term_means(code):
    assert cells(f"_ = {code}") == MEANINGS[code]


def test_a_bare_none_or_ellipsis_counts_only_in_files_that_use_an_array_library():
    for code in ("y = x[None]\n", "y = x[i, None]\n", "y = x[None, None]\n", "y = x[..., 0]\n", "y = x[0, ...]\n", "y = x[...]\n"):
        assert not Source(code).index.guides, code
        assert not Source("import os\n" + code).index.guides, code
        for imports in ("import numpy as np\n", "import torch\n", "from jax import numpy as jnp\n", "import pandas.api\n"):
            assert Source(imports + code).index.guides, (imports, code)


def test_what_needs_no_array_import():
    """A slice among several terms, or an explicit ``np.newaxis``, can only be array indexing."""
    for code in ("y = x[:, 0]\n", "y = x[..., :3]\n", "y = x[np.newaxis]\n", "y = x[0, np.newaxis]\n", "y = x[0, newaxis]\n"):
        assert Source(code).index.guides, code


def test_ordinals():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st", "111th",
    ]


# --------------------------------------------------------------------------- #
#  what is left alone
# --------------------------------------------------------------------------- #

PLAIN_PYTHON = [
    "_ = d['key']",
    "_ = xs[0]",
    "_ = xs[-1]",
    "_ = xs[i + 1]",
    "_ = s[1:]",
    "_ = s[:-1]",
    "_ = xs[::-1]",
    "_ = grid[r, c]",  # several indices, but nothing special about them
    "_ = cache[a, b]",
    # a bare None or ... means something else unless the file deals with arrays
    "_ = handlers[None]",
    "_ = d[a, None]",
    "_ = d[None, b]",
    "_ = expr[1, ...]",  # pyparsing: "one or more"
    "_ = expr[..., 3]",
    "_ = x[...]",
    "_ = table[lo:hi]",
    "_ = x[mask]",
    "_ = df.loc[df.a > 0, 'b']",
    # type expressions are not array indexing
    "Vec = Tuple[float, ...]",
    "Fn = Callable[..., int]",
    "Pair = tuple[int, ...]",
    "Opt = Optional[np.ndarray]",
    "T = typing.Dict[str, typing.Tuple[int, ...]]",
    "def f(x: tuple[int, ...]) -> Callable[..., int]: ...",
    "x: Dict[str, Tuple[int, ...]] = {}",
    "class A(Generic[T, ...]): ...",
    # inside an f-string nothing can be marked reliably
    "print(f'{x[:, 0]}')",
]


@pytest.mark.parametrize("code", PLAIN_PYTHON)
def test_ordinary_python_is_not_explained(code):
    assert not Source(code + "\n").index.guides


# --------------------------------------------------------------------------- #
#  the page
# --------------------------------------------------------------------------- #


class Chips(HTMLParser):
    """Collects the marked terms (``span.ax``) with their text, and the strip cells."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chips: List[dict] = []
        self.cells: List[dict] = []
        self.stack: List[str] = []
        self._open: List[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "span" and "ax" in classes:
            chip = {"kind": [c[3:] for c in classes if c.startswith("ik-")][0], "id": a["data-ix"], "text": ""}
            self.chips.append(chip)
            self._open.append(chip)
            self.stack.append("chip")
        elif tag == "span" and "ix-c" in classes:
            self.cells.append({"kind": [c[3:] for c in classes if c.startswith("ik-")][0], "id": a["data-ix"]})
            self.stack.append("span")
        elif tag in ("span", "div", "code", "i", "small", "b"):
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in ("span", "div", "code", "i", "small", "b") and self.stack:
            if self.stack.pop() == "chip":
                self._open.pop()

    def handle_data(self, data):
        for chip in self._open:
            chip["text"] += data


def chips_of(html: str) -> Chips:
    parser = Chips()
    parser.feed(html)
    return parser


class Balance(HTMLParser):
    """Every element that is opened must be closed, in order."""

    VOID = {"meta", "link", "br", "hr", "img", "input", "base", "col", "area", "wbr", "source", "track", "embed", "param"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[str] = []
        self.errors: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1:] or 'nothing'} at {self.getpos()}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def assert_balanced(html: str) -> None:
    parser = Balance()
    parser.feed(html)
    assert not parser.errors, parser.errors[:3]
    assert not parser.stack, f"never closed: {parser.stack[-5:]}"


def test_each_term_is_marked_and_explained_in_order():
    html = beautify("def f(img, n):\n    img = img[:, n:, ...]\n    return img\n")
    page = chips_of(html)
    assert [(c["kind"], c["text"]) for c in page.chips] == [("all", ":"), ("slice", "n:"), ("rest", "...")]
    assert [c["kind"] for c in page.cells] == ["all", "slice", "rest"]
    # every marked term points at the cell that explains it
    assert [c["id"] for c in page.chips] == [c["id"] for c in page.cells]
    assert "from <code>n</code> to the end" in html and "all remaining axes" in html


def test_the_code_itself_is_unchanged():
    code = "def f(img, n):\n    img = img[:, n:, ...]\n    return img\n"
    html, src = render_checked(code)
    assert not src.unrendered() and not src.duplicated()
    assert_balanced(html)
    # the marks add no text of their own to the code line
    line = re.search(r'<code class="c">(img .*?)</code>', html).group(1)
    assert re.sub(r"<[^>]+>", "", line) == "img = img[:, n:, ...]"


def test_reads_writes_and_deletes_are_told_apart():
    html = beautify("import numpy as np\ndef f(x):\n    y = x[:, 0]\n    x[:, 0] = 1\n    del x[..., None]\n")
    assert re.findall(r'<span class="ix-h">.*?<b>(\w+)</b>', html) == ["index", "assign", "delete"]


def test_the_same_subscript_twice_in_a_statement_is_explained_once():
    html = beautify("def f(x):\n    return x[:, 0] + x[:, 0]\n")
    page = chips_of(html)
    assert html.count('class="ix-r"') == 1
    assert len(page.chips) == 4 and len({c["id"] for c in page.chips}) == 2


def test_runs_of_none_share_one_cell_and_every_term_still_points_to_it():
    page = chips_of(beautify("def f(m):\n    return m[None, None, None, :]\n"))
    assert [c["kind"] for c in page.cells] == ["new", "all", "others"]
    assert [c["text"] for c in page.chips] == ["None", "None", "None", ":"]
    assert [c["id"] for c in page.chips][:3] == [page.cells[0]["id"]] * 3


def test_several_subscripts_in_one_statement_get_one_row_each_up_to_a_limit():
    body = " + ".join(f"a{i}[:, {i}]" for i in range(9))
    html = beautify(f"def f(*a):\n    return {body}\n")
    assert html.count('class="ix-r"') == 6
    assert "+ 3 more subscripts" in html
    assert len(chips_of(html).chips) == 18  # all of them are still marked in the code


def test_subscripts_in_headers_are_explained_above_the_block():
    html = beautify("def f(x):\n    if x[:, 0].any():\n        return 1\n    for r in x[::2, ...]:\n        pass\n")
    assert html.count('class="ix"') == 2
    strips = [m.start() for m in re.finditer(r'<div class="ix"', html)]
    guard, loop = html.index('<div class="guard'), html.index('<details class="box b-loop')
    assert strips[0] < guard < strips[1] < loop  # each explanation comes right before its statement


@pytest.mark.parametrize(
    "header",
    [
        "while x[:, 0].any():\n        break",
        "with ctx(x[:, 0]):\n        pass",
        "match x[:, 0]:\n        case 1:\n            pass",
        "if a:\n        pass\n    elif x[:, 0].any():\n        pass",
        "if a:\n        pass\n    elif b:\n        pass\n    elif x[..., 0].any():\n        pass",
    ],
)
def test_every_kind_of_header_gets_its_explanation(header):
    html = beautify(f"import numpy as np\ndef f(x, a, b, ctx):\n    {header}\n")
    assert html.count('class="ix"') == 1 and chips_of(html).chips


def test_an_else_followed_by_an_if_is_not_mistaken_for_an_elif():
    html = beautify("def f(x, a):\n    if a:\n        pass\n    else:\n        if x[:, 0].any():\n            pass\n")
    assert html.count('class="ix"') == 1  # explained once, above the nested if, not twice


def test_a_whole_if_chain_is_explained_above_it_in_order():
    html = beautify("import numpy as np\ndef f(x, a):\n    if x[:, 0].any():\n        pass\n    elif x[..., 1].any():\n        pass\n")
    assert html.count('class="ix"') == 1 and html.count('class="ix-r"') == 2
    assert html.index("x[:, 0]") < html.index("x[..., 1]")


def test_module_level_headers_are_only_marked_not_boxed_in_a_strip():
    html = beautify("for r in x[:, 0]:\n    pass\n")
    assert chips_of(html).chips and 'class="ix"' not in html


def test_the_toolbar_offers_the_switch_only_when_there_is_something_to_switch():
    assert 'id="b-index"' in beautify("y = x[:, 0]\n")
    assert 'id="b-index"' not in beautify("y = x[0]\n")


def test_the_switch_is_wired_up_in_the_assets():
    from python_beautifier.render.core import stylesheet

    assert ".no-index .ix" in stylesheet()
    js = (ROOT / "python_beautifier" / "assets" / "app.js").read_text(encoding="utf-8")
    assert '"#b-index"' in js and '"no-index"' in js and "data-ix" in js


def test_multi_line_subscripts_with_comments_stay_intact():
    code = "y = x[\n    :,  # every row\n    n:,  # skip the first n columns\n    ...,\n]\n"
    html, src = render_checked(code)
    assert not src.unrendered() and not src.duplicated()
    assert_balanced(html)
    page = chips_of(html)
    assert [c["kind"] for c in page.chips] == ["all", "slice", "rest"]
    assert "every row" in html and "skip the first n columns" in html  # the comments are still shown


def test_a_term_spanning_lines_is_marked_as_one_piece():
    code = "y = x[\n    a +\n    b :,\n    0,\n]\n"
    html, src = render_checked(code)
    assert not src.unrendered() and not src.duplicated()
    assert_balanced(html)
    assert len(chips_of(html).chips) == 2


def test_nested_subscripts_each_get_their_own_row_and_marks():
    html, src = render_checked("def f(x, idx):\n    return x[:, idx[:, 0]]\n")
    assert not src.unrendered() and not src.duplicated()
    assert_balanced(html)
    page = chips_of(html)
    assert html.count('class="ix-r"') == 2
    assert [c["text"] for c in page.chips if c["kind"] == "all"] == [":", ":"]


def test_hostile_subscripts_never_break_the_page():
    sources = [
        "x[*idx, :]\n",
        "x[(i := 1), :]\n",
        "x[:, [i for i in range(3)]]\n",
        "x[lambda: 1, :]\n",
        "x[:, ...][..., None][None]\n",
        "x[" + ", ".join([":"] * 300) + "]\n",
        "x[" + ", ".join(["None"] * 300) + "]\n",
        "x[..., ...]\n",
        "np.s_[1:3, ::2]\n",
        "class A:\n    y = x[:, 0]\n    def m(self, z=x[..., 0]):\n        return z[None, :]\n",
        "@deco(x[:, 0])\ndef f(): ...\n",
        "x[:, 0].y[..., 1].z[None]\n",
        "x[" + "y[" * 40 + ":, 0" + "]" * 40 + ", :]\n",
        "x[\u03c0:, \u00e4]\n",
    ]
    for code in sources:
        html, src = render_checked(code)
        assert not src.unrendered() and not src.duplicated(), code
        assert_balanced(html)


def test_a_second_ellipsis_is_reported_as_invalid():
    assert cells("_ = x[..., ...]")[1][2] == "invalid: only one `...` is allowed"


def test_render_items_ignores_marks_that_do_not_fit_its_range():
    src = Source("a = b[:, 0]\n")
    items = [it for it in src.items if it.kind != "newline"]
    tag = '<span class="m">'
    inside = ((1, 6), (1, 7), tag)  # the ":" itself
    sticking_out = ((1, 6), (1, 99), tag)  # would never close
    html = render_items(items, src.lines, (1, 0), (1, 11), 0, [inside, sticking_out])
    assert html.count("<span") == html.count("</span>") == 1


# --------------------------------------------------------------------------- #
#  the example file
# --------------------------------------------------------------------------- #


def test_the_tensor_example_page():
    code = EXAMPLE.read_text(encoding="utf-8")
    html, src = render_checked(code, "examples/tensor_indexing.py")
    assert not src.unrendered() and not src.duplicated()
    assert_balanced(html)
    page = chips_of(html)
    # every marked term is explained by a cell somewhere on the page
    cell_ids = {c["id"] for c in page.cells}
    assert {c["id"] for c in page.chips} - cell_ids == set()
    kinds = {c["kind"] for c in page.chips}
    assert {"all", "slice", "rest", "new", "pick", "take", "mask"} <= kinds
    assert "gather positions" in html and "keep where" in html and "assign" in html
