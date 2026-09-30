"""Shared fixtures and helpers."""
from __future__ import annotations

import html as htmllib
import re
import sys
from pathlib import Path
from typing import List, Set

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow ``pytest`` without installing the package
    sys.path.insert(0, str(ROOT))

from python_beautifier import beautify  # noqa: E402
from python_beautifier.model import build_module  # noqa: E402
from python_beautifier.render.core import Renderer  # noqa: E402
from python_beautifier.source import Source  # noqa: E402

SHOWCASE = ROOT / "examples" / "showcase.py"


@pytest.fixture(scope="session")
def showcase_text() -> str:
    return SHOWCASE.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def showcase_html(showcase_text: str) -> str:
    return beautify(showcase_text, filename="examples/showcase.py")


def classes_in(html: str) -> Set[str]:
    """Every CSS class used in the body markup of a rendered page."""
    body = html[html.index("<body"):]
    return {c for value in re.findall(r'class="([^"]*)"', body) for c in value.split()}


def render_checked(code: str, filename: str = "t.py"):
    """Render *code* and return ``(html, src)`` so tests can inspect the coverage bookkeeping."""
    src = Source(code, filename)
    html = Renderer(build_module(src)).render()
    return html, src


def visible_text(html: str) -> str:
    """The text a reader would see (tags, scripts and styles removed)."""
    body = html[html.index("<body"):] if "<body" in html else html
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
    return htmllib.unescape(re.sub(r"<[^>]+>", "", body))


def tricky_snippets() -> List[str]:
    """Small programs that exercise unusual syntax."""
    return [
        "",
        "\n\n",
        "# only a comment\n",
        '"""Only a docstring."""\n',
        "x = 1\n",
        "def f(a, /, b, *, c=1, **kw): return a + b + c\n",
        "async def g():\n    async with a as b, c as d:\n        async for i in b:\n            await d(i)\n    return [x async for x in y]\n",
        "class A(Base, metaclass=M):\n    x: int = 1\n    def m(self): ...\n    @property\n    def p(self) -> int: return 1\n",
        "if (n := len(a)) > 10: print(n)\n",
        "x = lambda a, *b, c=1, **d: (a, b, c, d)\n",
        'f"{a!r:>{w}} and {b=} {{literal}}"\n',
        "try:\n    pass\nexcept* ValueError as eg:\n    raise\n",
        "while 1:\n    if x: break\nelse:\n    pass\n",
        "a = 1; b = 2; c = 3\n",
        "value = (1 +\n         2 +  # comment inside\n         3)\n",
        "total = 1 + \\\n    2\n",
        "def outer():\n    x = 1\n    def inner():\n        nonlocal x\n        x += 1\n    global G\n    return inner\n",
        "match command.split():\n    case [\"go\", direction] if direction in DIRS:\n        pass\n    case [\"drop\", *objects]:\n        pass\n    case {\"k\": v, **rest}:\n        pass\n    case Point(x=0, y=0) | None:\n        pass\n    case _:\n        pass\n",
        "with (open('a') as f, open('b') as g):\n    pass\n",
        "names = {k: v for k, v in zip(a, b) if k}\nprint(*names, sep='')\n",
        "\tx = 1\n" if False else "if True:\n\tx = 1\n\tif x:\n\t\ty = 2\n",
        "π = 3.14159\nnaïve = 'ünïcode ✓'\n",
        "from __future__ import annotations\nfrom . import sibling\nfrom .. import parent as p\nimport os.path as osp, sys\n",
        "@decorator(arg=1)\n@other.attr\nclass D:\n    '''Doc.'''\n",
        "def deco(fn):\n    @wraps(fn)\n    def wrapper(*a, **k):\n        return fn(*a, **k)\n    return wrapper\n",
        "x = [\n    1,  # one\n    2,  # two\n]\n",
        "def f():\n    '''Doc only.'''\n",
        "def f():\n    pass  # trailing on pass\n",
        "assert x, 'message'\ndel a[0], b\nraise SystemExit from None\n",
        "print('no newline at end')",
        "for i in range(3):\n    for j in range(3):\n        if i == j:\n            continue\n        print(i, j)\nelse:\n    print('done')\n",
    ]
