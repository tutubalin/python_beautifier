"""Alternative model entrypoints are found by their layer calls and drawn as inline SVG."""
from __future__ import annotations

from html.parser import HTMLParser
import re

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


def test_crossing_skip_connections_use_distinct_colors_and_lanes():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 8)
        self.d = nn.Linear(8, 8)
    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        merged = a + b
        c = self.c(merged)
        d = self.d(c)
        return c + d
'''
    html = beautify(code)
    paths = re.findall(r'<path class="nn-edge nn-skip nn-flow-\d+" d="([^"]+)"', html)
    rails = [re.search(r"H (\d+) V", path).group(1) for path in paths]
    assert len(paths) == 4
    assert len(set(rails)) == 2  # two independent merges reuse the same two clear lanes
    assert set(rails[:2]) == set(rails[2:])
    assert rails[0] != rails[1] and rails[2] != rails[3]
    for color in range(4):
        assert f'class="nn-edge nn-skip nn-flow-{color}"' in html
        assert f'id="nn-0-Net-arrow-skip-{color}"' in html
        assert f".nn-edge.nn-skip.nn-flow-{color}" in html


def test_merge_schema_svg_marks_a_skip_connection():
    from pathlib import Path

    fixture = Path(__file__).resolve().parents[1] / "examples" / "vision_model.py"
    html = beautify(fixture.read_text(encoding="utf-8"), filename="vision_model.py")
    assert 'class="nn-svg-node nn-svg-merge"' in html
    assert "nn-edge nn-skip" in html
    assert 'role="img"' in html and "INPUT" in html and "OUTPUT" in html


def test_modulelist_comprehension_and_custom_non_forward_method_calls_remain_visible():
    code = '''
from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
    def forward_kv_extract(self, x, cache):
        return self.proj(x), cache
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
    def forward(self, x, cache):
        for i, block in enumerate(self.blocks):
            x, cache = block.forward_kv_extract(x, cache)
        return x
'''
    schemas = analyze(Source(code))
    schema = next(item for item in schemas.values() if item.name == "Net")
    block_calls = [step for step in schema.steps if "blocks[*].forward_kv_extract" in step.name]
    assert len(block_calls) == 1  # one representative iteration, not three fabricated copies
    assert block_calls[0].kind == "Block"
    assert block_calls[0].out_shape == ("…", "?")
    assert any("multiple outputs are not separated" in note for note in schema.notes)
    assert any("shown once" in note for note in schema.notes)


def test_nested_custom_module_attributes_are_detected_and_traced():
    code = '''
from torch import nn
class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.holder = Holder()
    def encode(self, x):
        return self.holder.proj(x)
'''
    schemas = analyze(Source(code))
    schema = next(item for item in schemas.values() if item.name == "Net")
    assert [route.method_name for route in schema.routes] == ["encode"]
    assert [(step.name, step.kind) for step in schema.steps] == [("holder.proj", "Linear")]
    assert schema.output_shapes == [("…", "4")]


def test_docstring_layout_labels_are_not_misread_as_tensor_dimensions():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
    def forward_kv_extract(self, img):
        """img has layout [ref, img], not a literal tensor shape."""
        return self.proj(img)
'''
    schema = _schema(code)
    assert schema.input_shape == ("…", "4")
    assert schema.output_shapes == [("…", "4")]


def test_functional_normalization_is_detected_from_contents_and_stays_shape_preserving():
    code = '''
import torch
from torch import nn
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor[B, T, D]):
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return x * rms * self.scale
'''
    schemas = analyze(Source(code))
    schema = next(item for item in schemas.values() if item.name == "RMSNorm")
    assert [step.kind for step in schema.steps] == ["Mean", "Rsqrt"]
    assert schema.input_shape == ("B", "T", "D")
    assert schema.output_shapes == [("B", "T", "D")]
    assert not any("Declared modules not observed" in note for note in schema.notes)


def test_example_documents_nonstandard_entrypoints():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "examples" / "vision_model.py").read_text(encoding="utf-8")
    schema = next(s for s in analyze(Source(source)).values() if s.name == "KVProjection")
    assert [r.method_name for r in schema.routes] == ["forward_kv_extract", "forward_kv_cached"]
    assert all(r.steps for r in schema.routes)
