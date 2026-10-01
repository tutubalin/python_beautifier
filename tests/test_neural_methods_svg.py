"""Alternative model entrypoints are found by their layer calls and drawn as inline SVG."""
from __future__ import annotations

from html.parser import HTMLParser

from python_beautifier import beautify
from python_beautifier.neural import analyze
from python_beautifier.source import Source


TWO_PATHS = '''
import torch
from torch import nn
class CacheNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4)
        self.out = nn.Linear(4, 2)
    def forward_kv_extract(self, hidden_states):
        return self.out(self.proj(hidden_states))
    def forward_kv_cached(self, hidden_states):
        return self.proj(hidden_states)
'''


class SvgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.svg = 0
        self.markers = []
        self.titles = []
        self.paths = 0
        self.names = []
        self.details = 0
        self.in_svg = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "svg" and "nn-svg" in attrs.get("class", "").split():
            self.svg += 1
            self.in_svg = True
        elif self.in_svg and tag == "marker":
            self.markers.append(attrs.get("id"))
        elif self.in_svg and tag == "path":
            self.paths += 1
        elif tag == "h5" and "nn-route-title" in attrs.get("class", ""):
            self.names.append("")
        elif tag == "details" and "nn-detail" in attrs.get("class", ""):
            self.details += 1

    def handle_endtag(self, tag):
        if tag == "svg":
            self.in_svg = False

    def handle_data(self, data):
        if self.names and not self.names[-1]:
            self.names[-1] = data.strip()


def _schema(code=TWO_PATHS):
    schemas = analyze(Source(code))
    assert len(schemas) == 1
    return next(iter(schemas.values()))


def test_multiple_forward_named_methods_are_detected_from_their_layer_calls():
    schema = _schema()
    assert schema.method_name == "forward_kv_extract"
    assert [r.method_name for r in schema.routes] == ["forward_kv_extract", "forward_kv_cached"]
    assert [len(r.steps) for r in schema.routes] == [2, 1]
    assert [r.output_shapes for r in schema.routes] == [[("…", "2")], [("…", "4")]]


def test_an_arbitrary_method_name_is_found_from_what_it_does():
    code = '''
from torch import nn
class Model:
    def __init__(self):
        self.embedding = nn.Embedding(100, 12)
        self.projection = nn.Linear(12, 3)
    def encode_tokens(self, ids):
        return self.projection(self.embedding(ids))
'''
    schema = _schema(code)
    assert schema.detection == "lookalike"
    assert [r.method_name for r in schema.routes] == ["encode_tokens"]
    assert [step.kind for step in schema.steps] == ["Embedding", "Linear"]
    assert schema.output_shapes == [("…", "12", "3")]


def test_helper_methods_called_by_a_content_discovered_forward_are_not_duplicate_routes():
    code = '''
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4)
    def _project(self, x):
        return self.proj(x)
    def run_encoder(self, x):
        return self._project(x)
'''
    schema = _schema(code)
    # The caller is discovered by traversing method content; its helper is inlined to the layer.
    assert [r.method_name for r in schema.routes] == ["run_encoder"]
    assert [step.name for step in schema.steps] == ["proj"]
    assert schema.output_shapes == [("…", "4")]


def test_schema_contains_one_inline_svg_per_execution_method_and_an_expandable_trace():
    html = beautify(TWO_PATHS)
    parser = SvgParser()
    parser.feed(html)
    assert parser.svg == 2
    assert len(parser.markers) == 2 and len(set(parser.markers)) == 2
    assert parser.paths >= 4  # layer links plus input and output arrows
    assert parser.details == 2
    assert 'class="nn-route-title"><code>forward_kv_extract()' in html
    assert 'class="nn-route-title"><code>forward_kv_cached()' in html
    assert "[…, 8]" in html and "[…, 4]" in html and "[…, 2]" in html
    assert "method contents" in html and "model code is never run" in html


def test_merge_schema_svg_marks_a_skip_connection():
    from pathlib import Path

    fixture = Path(__file__).resolve().parents[1] / "examples" / "vision_model.py"
    html = beautify(fixture.read_text(encoding="utf-8"), filename="vision_model.py")
    assert 'class="nn-svg-node nn-svg-merge"' in html
    assert "nn-edge nn-skip" in html
    assert 'role="img"' in html and "INPUT" in html and "OUTPUT" in html


def test_example_documents_nonstandard_entrypoints():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "examples" / "vision_model.py").read_text(encoding="utf-8")
    schema = next(s for s in analyze(Source(source)).values() if s.name == "KVProjection")
    assert [r.method_name for r in schema.routes] == ["forward_kv_extract", "forward_kv_cached"]
    assert all(r.steps for r in schema.routes)
