"""Static architecture inference for torch.nn.Module-like classes."""
from __future__ import annotations

from pathlib import Path

import pytest

from python_beautifier import beautify
from python_beautifier.neural import Schema, analyze, format_shape
from python_beautifier.source import Source
from conftest import render_checked

ROOT = Path(__file__).resolve().parents[1]


def schemas(code: str):
    src = Source(code)
    return src, analyze(src)


def _schema(code: str, name: str = "Net") -> Schema:
    _, found = schemas(code)
    assert name in [s.name for s in found.values()]
    return next(s for s in found.values() if s.name == name)


@pytest.mark.parametrize(
    "imports,base",
    [
        ("import torch.nn as nn", "nn.Module"),
        ("import torch", "torch.nn.Module"),
        ("from torch import nn as layers", "layers.Module"),
        ("from torch.nn import Module as Base", "Base"),
        ("from torch.nn.modules.module import Module", "Module"),
    ],
)
def test_all_common_import_aliases_detect_real_pytorch_bases(imports, base):
    code = f"{imports}\nclass Net({base}):\n    def forward(self, x):\n        return x\n"
    assert _schema(code).detection == "pytorch"


def test_torch_imported_as_alias():
    code = "import torch as th\nclass Net(th.nn.Module):\n    def forward(self, x): return x\n"
    assert _schema(code).detection == "pytorch"


def test_indirect_local_inheritance_is_followed():
    code = '''
from torch import nn
class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x)
class Intermediate(Base):
    pass
class Net(Intermediate):
    pass
'''
    _, found = schemas(code)
    derived = next(s for s in found.values() if s.name == "Net")
    assert derived.detection == "inherited"
    assert [(step.name, step.out_shape) for step in derived.steps] == [("fc", ("…", "4"))]


def test_child_can_override_an_inherited_forward():
    code = '''
import torch.nn as nn
class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x): return self.fc(x)
class Net(Base):
    def forward(self, x): return self.fc(x).relu()
'''
    schema = _schema(code)
    assert schema.steps[0].kind == "Linear"
    assert schema.output_shapes == [("…", "4")]


def test_structural_lookalike_uses_layer_names_even_without_torch_imports():
    code = '''
class Net:
    def __init__(self):
        self.proj = Linear(8, 4)
        self.act = ReLU()
    def forward(self, x):
        return self.act(self.proj(x))
'''
    schema = _schema(code)
    assert schema.detection == "lookalike"
    assert [s.kind for s in schema.steps] == ["Linear", "ReLU"]
    assert schema.output_shapes == [("…", "4")]


def test_plain_classes_are_not_misclassified():
    for code in (
        "class Net:\n    def forward(self, x): return x\n",
        "class Net:\n    def __init__(self): self.x = Linear(8, 4)\n",
        "class Net:\n    def __init__(self): self.x = object()\n    def run(self, x): return self.x(x)\n",
    ):
        assert not analyze(Source(code)), code


def test_a_detected_base_without_a_layer_calling_method_gets_an_explanation_not_a_fake_graph():
    schema = _schema("from torch import nn\nclass Net(nn.Module): pass\n")
    assert not schema.steps
    assert "No method body calling a declared layer" in schema.notes[0]


def test_conv_pool_flatten_linear_shapes_propagate_from_forward_docstring():
    code = '''
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 16 * 16, 10)
    def forward(self, x):
        """Input x has shape (B, 3, 32, 32)."""
        return self.fc(torch.flatten(self.pool(self.conv(x)), 1))
'''
    schema = _schema(code)
    assert schema.input_shape == ("B", "3", "32", "32")
    assert [(s.kind, s.in_shape, s.out_shape) for s in schema.steps] == [
        ("Conv2d", ("B", "3", "32", "32"), ("B", "8", "32", "32")),
        ("MaxPool2d", ("B", "8", "32", "32"), ("B", "8", "16", "16")),
        ("Flatten", ("B", "8", "16", "16"), ("B", "2048")),
        ("Linear", ("B", "2048"), ("B", "10")),
    ]
    assert schema.output_shapes == [("B", "10")]


def test_explicit_tensor_shape_annotation_is_used():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 3)
    def forward(self, x: Tensor[B, 3, 4]):
        return self.fc(x.flatten(1))
'''
    schema = _schema(code)
    assert schema.input_shape == ("B", "3", "4")
    assert schema.output_shapes == [("B", "3")]


def test_sequential_is_expanded_and_connected_in_order():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x): return self.layers(x)
'''
    schema = _schema(code)
    assert [s.name for s in schema.steps] == ["layers.0", "layers.1", "layers.2"]
    assert [s.incoming for s in schema.steps] == [("input",), ("n1",), ("n2",)]
    assert [s.out_shape for s in schema.steps] == [("…", "8"), ("…", "8"), ("…", "2")]
    assert not schema.notes


def test_module_list_iteration_is_expanded_once_and_marked_as_a_loop():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 2)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
'''
    schema = _schema(code)
    assert [s.name for s in schema.steps] == ["layers.0", "layers.1", "layers.2"]
    assert any("shown once" in note for note in schema.notes)


def test_direct_index_into_a_sequential_uses_only_that_child():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x): return self.layers[2](x)
'''
    schema = _schema(code)
    assert [(s.name, s.kind) for s in schema.steps] == [("layers.2", "Linear")]


def test_elementwise_merge_shows_both_incoming_connections():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(8, 8)
        self.right = nn.Linear(8, 8)
        self.out = nn.Linear(8, 2)
    def forward(self, x):
        a = self.left(x)
        b = self.right(x)
        return self.out(a + b)
'''
    schema = _schema(code)
    merge = next(s for s in schema.steps if s.merge)
    assert merge.incoming == ("n1", "n2")
    assert schema.steps[-1].incoming == (merge.ident,)


def test_if_branches_are_shown_as_alternatives_not_a_claimed_sequence():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4)
        self.b = nn.Linear(8, 4)
        self.out = nn.Linear(4, 2)
    def forward(self, x, flag):
        if flag:
            x = self.a(x)
        else:
            x = self.b(x)
        return self.out(x)
'''
    schema = _schema(code)
    assert [s.name for s in schema.steps] == ["a", "b", "out"]
    assert schema.steps[-1].incoming == ("n1", "n2")
    assert any("only one path" in note for note in schema.notes)


def test_shapes_stay_symbolic_when_spatial_size_is_unknown():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x): return self.pool(self.conv(x))
'''
    schema = _schema(code)
    assert schema.input_shape == ("…", "3", "H", "W")
    assert schema.steps[0].out_shape == ("…", "16", "H", "W")
    assert schema.steps[1].out_shape == ("…", "16", "Hout", "Wout")


def test_lookalike_without_a_real_model_library_is_explicitly_uncertain():
    html = beautify("class Net:\n    def __init__(self): self.fc = Linear(8, 4)\n    def forward(self, x): return self.fc(x)\n")
    assert "module-like pattern" in html
    assert "model code is never run" in html
    assert "[\u2026, 8]" in html and "[\u2026, 4]" in html


def test_non_module_class_cards_do_not_get_the_schema_section():
    html = beautify("class Plain:\n    def run(self, x): return x\n")
    assert "Model schema" not in html


def test_factories_custom_residual_blocks_and_wrappers_are_expanded_without_execution():
    code = '''
import torch
from torch import nn

def conv(a, b, **kw):
    return nn.Conv2d(a, b, 3, padding=1, **kw)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(conv(in_channels, out_channels), nn.ReLU(), conv(out_channels, out_channels))
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))

def Encoder(channels=4):
    return nn.Sequential(conv(3, 8), Block(8, 8), conv(8, channels, stride=2))

class Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(4)
'''
    found = analyze(Source(code))
    schemas_by_name = {schema.name: schema for schema in found.values()}

    # Functional-only modules are selected from method contents; factory returns and
    # custom residual blocks are expanded from syntax without importing or running code.
    assert [step.kind for step in schemas_by_name["Clamp"].steps] == ["Tanh"]
    block = schemas_by_name["Block"]
    assert [step.kind for step in block.steps] == ["Conv2d", "ReLU", "Conv2d", "Conditional layer", "Add", "ReLU"]
    assert block.output_shapes == [("…", "out_channels", "H", "W")]

    factory = schemas_by_name["Encoder"]
    wrapper = schemas_by_name["Wrap"]
    for schema in (factory, wrapper):
        assert [step.kind for step in schema.steps].count("Conv2d") == 4
        assert "Identity" in [step.kind for step in schema.steps]
        assert schema.output_shapes == [("…", "4", "ceil(H/2)", "ceil(W/2)")]
    assert factory.detection == "factory"
    assert wrapper.routes[0].kind == "attribute"
    assert "no own forward method" in wrapper.routes[0].notes[0]

    html = beautify(code)
    assert "sequential factory" in html
    assert "Model schema" in html
    assert "ceil(H/2)" in html


def test_wrapper_reports_conditional_factory_replacements_instead_of_claiming_one_architecture():
    code = '''
from torch import nn

def BaseEncoder():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))

def FastEncoder():
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))

class Model(nn.Module):
    def __init__(self, fast=False):
        super().__init__()
        self.encoder = BaseEncoder()
        if fast:
            self.encoder = FastEncoder()
'''
    found = {schema.name: schema for schema in analyze(Source(code)).values()}
    model = found["Model"]
    assert model.routes[0].output_shapes == [("…", "4", "H", "W")]
    assert any("replaced by FastEncoder" in note and "when fast" in note for note in model.routes[0].notes)
    assert found["FastEncoder"].detection == "factory"
    assert found["FastEncoder"].output_shapes == [("…", "8", "H", "W")]


def test_realistic_example_schema():
    src = Source((ROOT / "examples" / "vision_model.py").read_text(encoding="utf-8"))
    found = analyze(src)
    schema = next(s for s in found.values() if s.name == "TinyVisionNet")
    assert schema.detection == "inherited"
    assert [s.kind for s in schema.steps] == ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU", "Conv2d", "Add", "AdaptiveAvgPool2d", "Flatten", "Linear"]
    assert schema.steps[0].in_shape == ("B", "3", "32", "32")
    assert schema.steps[0].out_shape == ("B", "16", "32", "32")
    assert schema.steps[-1].out_shape == ("B", "classes")
    assert not schema.notes


def test_model_diagram_is_rendered_in_class_card_and_source_integrity_is_preserved():
    code = '''
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x): return self.fc(x)
'''
    html = beautify(code)
    assert 'class="sec nn-schema"' in html
    assert "Model schema" in html
    schema_html = html[html.index('class="sec nn-schema"') : html.index('</section>', html.index('class="sec nn-schema"'))]
    assert "out_features=4" in schema_html and "nn.Linear" not in schema_html
    assert "Linear" in html and "out_features=4" in html
    assert 'class="nn-line" href="#L' in html
    checked, src = render_checked(code)
    assert "nn-schema" in checked and not src.unrendered() and not src.duplicated()


def test_outputs_and_shapes_use_safe_human_readable_markup():
    schema = _schema("from torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n    def forward(self, x): return self.fc(x)\n")
    assert format_shape(schema.steps[0].out_shape) == "[…, 2]"
    html = beautify("from torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n    def forward(self, x): return self.fc(x)\n")
    assert '<ul class="nn-notes"' not in html


def test_custom_imported_child_modules_are_shown_as_opaque_blocks():
    code = """
from torch import nn
from my_layers import FeatureBlock
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureBlock(width=32)
        self.head = nn.Linear(32, 4)
    def forward(self, x): return self.head(self.features(x))
"""
    schema = _schema(code)
    assert [s.kind for s in schema.steps] == ["FeatureBlock", "Linear"]
    assert schema.steps[0].out_shape == ("…", "?")
    assert schema.steps[1].out_shape == ("…", "4")


def test_torch_functional_reshape_permute_and_pool_shapes():
    code = """
import torch
from torch import nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, 2)
        x = torch.permute(x, (0, 2, 3, 1))
        return torch.reshape(x, (x.shape[0], -1))
"""
    schema = _schema(code)
    assert schema.steps[0].out_shape == ("…", "8", "H", "W")
    assert schema.steps[-1].out_shape == ("…", "features")
    assert schema.output_shapes == [("…", "features")]


def test_incompatible_static_linear_input_is_flagged_instead_of_silently_accepted():
    code = """
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        \"\"\"Input x has shape (B, 6).\"\"\"
        return self.fc(x)
"""
    schema = _schema(code)
    assert schema.steps[-1].out_shape == ("B", "4")
    assert any("expects 8 input features" in note for note in schema.notes)


def test_realistic_model_example_renders_without_lost_or_repeated_source():
    from conftest import render_checked

    code = (ROOT / "examples" / "vision_model.py").read_text(encoding="utf-8")
    html, src = render_checked(code, "examples/vision_model.py")
    assert 'class="sec nn-schema"' in html
    assert not src.unrendered() and not src.duplicated()


def test_renamed_torch_layer_import_retains_its_known_shape_rule():
    code = """
from torch.nn import Module as Base, Linear as Dense
class Net(Base):
    def __init__(self):
        super().__init__()
        self.fc = Dense(8, 4)
    def forward(self, x): return self.fc(x)
"""
    schema = _schema(code)
    assert schema.steps[0].kind == "Linear"
    assert schema.steps[0].params == "in_features=8, out_features=4"
    assert schema.steps[0].out_shape == ("…", "4")


def test_torch_cat_draws_a_join_and_combines_the_joined_dimension():
    code = """
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4)
        self.b = nn.Linear(8, 3)
        self.out = nn.Linear(7, 2)
    def forward(self, x):
        left = self.a(x)
        right = self.b(x)
        return self.out(torch.cat((left, right), dim=-1))
"""
    schema = _schema(code)
    concat = next(s for s in schema.steps if s.kind == "Concat")
    assert concat.incoming == ("n1", "n2")
    assert concat.out_shape == ("…", "7")
    assert schema.steps[-1].in_shape == ("…", "7")
    assert schema.output_shapes == [("…", "2")]
