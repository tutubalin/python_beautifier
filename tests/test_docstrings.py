from python_beautifier.docstrings import first_sentence, inline_html, parse

GOOGLE = '''Fetch a thing from the network.

    Longer description that spans
    several lines.

    - first bullet
    - second bullet

    Args:
        url (str): The address to fetch.
        timeout (float, optional): Seconds to wait. Defaults to 3.
        headers (dict[str, str]): Extra headers,
            wrapped onto a second line.
        *args: Ignored.
        **kwargs: Passed through.

    Returns:
        bytes: The response body.

    Raises:
        ValueError: If the URL is malformed.
        TimeoutError: If the server is too slow.

    Example:
        >>> fetch("http://x")
        b'...'

    Note:
        Not thread safe.
    '''

NUMPY = """Compute the mean.

Some longer text.

Parameters
----------
a : array_like
    Input data.
axis : int, optional
    Axis along which to average.
x, y : float
    Two coordinates.

Returns
-------
mean : ndarray
    The average.

Raises
------
ValueError
    If empty.

Notes
-----
Uses pairwise summation.
"""

SPHINX = """Do the thing.

:param int count: How many.
:param name: The name
    continued here.
:type name: str
:returns: a result
:rtype: bool
:raises KeyError: when missing
"""

EPYTEXT = """Short summary.

@param a: first
@type a: int
@return: the sum
@rtype: int
@raise ValueError: on bad input
"""


def test_google_params_and_sections():
    d = parse(GOOGLE)
    assert d.style == "google"
    assert d.summary == "Fetch a thing from the network."
    names = [f.name for f in d.params]
    assert names == ["url", "timeout", "headers", "*args", "**kwargs"]
    url = d.param("url")
    assert url.type == "str" and url.desc == "The address to fetch."
    t = d.param("timeout")
    assert t.type == "float" and t.optional
    assert d.param("headers").type == "dict[str, str]"
    assert d.param("headers").desc == "Extra headers, wrapped onto a second line."
    assert d.param("args").desc == "Ignored."
    assert d.returns[0].type == "bytes" and d.returns[0].desc == "The response body."
    assert [r.name for r in d.raises] == ["ValueError", "TimeoutError"]
    kinds = [s.kind for s in d.sections]
    assert kinds == ["example", "note"]
    assert d.sections[0].blocks[0].kind == "doctest"
    assert any(b.kind == "ul" and len(b.items) == 2 for b in d.blocks)


def test_numpy():
    d = parse(NUMPY)
    assert d.style == "numpy"
    assert [f.name for f in d.params] == ["a", "axis", "x", "y"]
    assert d.param("a").type == "array_like"
    assert d.param("axis").optional and d.param("axis").type == "int"
    assert d.param("y").type == "float"
    assert d.returns[0].name == "mean" and d.returns[0].type == "ndarray"
    assert d.raises[0].name == "ValueError" and d.raises[0].desc == "If empty."
    assert d.sections[0].kind == "note"


def test_sphinx():
    d = parse(SPHINX)
    assert d.style == "sphinx"
    assert d.param("count").type == "int"
    assert d.param("name").type == "str"
    assert d.param("name").desc == "The name continued here."
    assert d.returns[0].type == "bool" and d.returns[0].desc == "a result"
    assert d.raises[0].name == "KeyError"
    assert d.summary == "Do the thing."


def test_epytext():
    d = parse(EPYTEXT)
    assert d.param("a").type == "int" and d.param("a").desc == "first"
    assert d.returns[0].type == "int"
    assert d.raises[0].name == "ValueError"


def test_plain_and_empty():
    assert not parse(None)
    assert not parse("   \n ")
    d = parse("Just a line.")
    assert d.summary == "Just a line." and not d.params and not d.has_details


def test_literal_block_and_admonition():
    d = parse(
        """Summary.

        Example::

            x = 1
            y = 2

        .. note:: Be careful.

        .. versionadded:: 3.4
        """
    )
    kinds = [b.kind for b in d.blocks]
    assert "code" in kinds and "note" in kinds


def test_google_returns_without_type():
    d = parse("Do.\n\nReturns:\n    True if it worked, otherwise False.\n")
    assert d.returns[0].type == "" and d.returns[0].desc.startswith("True if")


def test_inline_markup():
    out = inline_html("Use `foo` and ``bar`` or :class:`Baz`. **bold** and *em* <x>", lambda n: "#d-Baz" if n == "Baz" else None)
    assert '<code class="ic">foo</code>' in out and '<code class="ic">bar</code>' in out
    assert 'href="#d-Baz"' in out
    assert "<strong>bold</strong>" in out and "<em>em</em>" in out
    assert "&lt;x&gt;" in out
    assert "<script" not in inline_html("<script>alert(1)</script>")


def test_inline_does_not_mangle_star_args():
    assert inline_html("takes *args and **kwargs") == "takes *args and **kwargs"


def test_first_sentence():
    assert first_sentence("One. Two.") == "One."
    assert first_sentence("x" * 300).endswith("\u2026")


# --- unindented code blocks and text after sections -------------------------

USAGE = '''Heap queue.

Usage:

heap = []            # creates an empty heap
heappush(heap, item) # pushes a new item on the heap
item = heappop(heap) # pops the smallest item

Our API differs as follows:

- We use 0-based indexing.
'''


def test_unindented_usage_body_is_code_and_later_text_is_kept():
    doc = parse(USAGE)
    usage = doc.sections[0]
    assert usage.title == "Usage"
    assert [b.kind for b in usage.blocks] == ["code"]
    assert "heappush(heap, item)" in usage.blocks[0].text
    after = doc.sections[1]
    assert after.kind == "plain" and after.blocks[0].text.startswith("Our API differs")
    assert after.blocks[1].kind == "ul"


def test_multiline_paragraph_that_is_valid_python_is_code():
    doc = parse("Summary.\n\nx = compute(1)\ny = compute(2)\n")
    assert [b.kind for b in doc.blocks] == ["p", "code"]


def test_prose_spanning_lines_stays_prose():
    doc = parse("Summary.\n\nThis is a sentence that\nwraps over two lines.\n")
    assert [b.kind for b in doc.blocks] == ["p", "p"]
    assert doc.blocks[1].text == "This is a sentence that wraps over two lines."


def test_google_section_ends_where_indentation_returns():
    doc = parse("Do things.\n\nArgs:\n    x (int): the thing\n        continued here.\n    y: other\n\nReturns:\n    int: result\n\nExtra words at the end.\n")
    assert [(p.name, p.type, p.desc) for p in doc.params] == [("x", "int", "the thing continued here."), ("y", "", "other")]
    assert [(r.type, r.desc) for r in doc.returns] == [("int", "result")]
    assert doc.sections[-1].kind == "plain" and doc.sections[-1].blocks[0].text == "Extra words at the end."
