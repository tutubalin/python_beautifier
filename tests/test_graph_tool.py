from __future__ import annotations

from pathlib import Path

from python_beautifier import beautify_graph
from python_beautifier.graph_cli import main


MODEL = '''
from torch import nn
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4)
    def forward(self, x):
        return self.proj(x)
'''


def test_graph_only_api_returns_a_focused_self_contained_page():
    html = beautify_graph(MODEL, filename="model.py")
    assert "SmallNet" in html and "Model schema" in html
    assert "Graph Explorer" in html or "· Graphs" in html
    assert "Source walkthrough" not in html
    assert 'id="pb-source"' not in html
    assert "nn-svg" in html
    assert "Toggle theme" in html


def test_graph_only_page_has_a_friendly_empty_state():
    html = beautify_graph("def helper(x): return x\n", filename="plain.py")
    assert "No statically traceable neural model schemas" in html
    assert "Model code is analyzed, never executed" in html


def test_graph_only_cli_writes_a_separate_compact_output(tmp_path: Path):
    source = tmp_path / "network.py"
    output = tmp_path / "graphs.html"
    source.write_text(MODEL, encoding="utf-8")
    assert main([str(source), "-o", str(output), "-q"]) == 0
    page = output.read_text(encoding="utf-8")
    assert "SmallNet" in page and "Model schema" in page


def test_graph_only_page_includes_drag_navigation_and_route_index():
    html = beautify_graph(MODEL, filename="model.py")
    assert "pointerdown" in html and "scrollLeft" in html
    assert 'href="#nn-0-SmallNet"' in html
    assert 'id="nn-0-SmallNet"' in html
