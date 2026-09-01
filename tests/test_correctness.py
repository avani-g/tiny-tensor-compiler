import torch
import torch.nn as nn
import torch.fx as fx

from torch.fx.passes.shape_prop import ShapeProp

from compiler.frontend import fx_to_ir
from compiler.passes import (
    PassManager,
    algebraic_simplification,
    constant_folding,
    dead_code_elimination,
)
from compiler.fusion import fuse_linear_relu
from compiler.backend import execute


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        x = self.linear(x)
        x = x + 0
        x = x * 1
        x = torch.relu(x)

        return x


def compile_model(model, example_input):
    traced = fx.symbolic_trace(model)

    ShapeProp(traced).propagate(example_input)

    graph = fx_to_ir(traced)

    manager = PassManager()
    manager.add_pass(algebraic_simplification)
    manager.add_pass(constant_folding)
    manager.add_pass(dead_code_elimination)
    manager.add_pass(fuse_linear_relu)

    return manager.run(graph)


def test_compiled_output_matches_pytorch():
    model = TinyModel()
    model.eval()

    test_input = torch.randn(32, 10)

    optimized_graph = compile_model(
        model,
        test_input,
    )

    with torch.no_grad():
        pytorch_output = model(test_input)

        compiled_output = execute(
            optimized_graph,
            model,
            test_input,
        )

    assert torch.allclose(
        pytorch_output,
        compiled_output,
        rtol=1e-5,
        atol=1e-6,
    )