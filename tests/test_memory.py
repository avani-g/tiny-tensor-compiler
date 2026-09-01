import torch
import torch.nn as nn
import torch.fx as fx

from torch.fx.passes.shape_prop import ShapeProp

from compiler.ir import IRNode

from compiler.frontend import fx_to_ir
from compiler.backend import execute


def test_tensor_size_bytes():
    node = IRNode(
        name="x",
        op="input",
        inputs=[],
        shape=[32, 20],
        dtype="torch.float32",
    )

    assert node.num_elements() == 640
    assert node.size_bytes() == 2560

from compiler.ir import IRNode, IRGraph


def test_peak_memory():
    graph = IRGraph()

    graph.add_node(
        IRNode(
            "x",
            "input",
            [],
            shape=[10],
            dtype="torch.float32",
        )
    )

    graph.add_node(
        IRNode(
            "a",
            "relu",
            ["x"],
            shape=[20],
            dtype="torch.float32",
        )
    )

    graph.add_node(
        IRNode(
            "b",
            "relu",
            ["a"],
            shape=[30],
            dtype="torch.float32",
        )
    )

    graph.add_node(
        IRNode(
            "output",
            "return",
            ["b"],
        )
    )

    assert graph.compute_peak_memory() == 200

def test_liveness_aware_execution_reduces_retained_memory():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)

        def forward(self, x):
            x = self.linear(x)
            x = x + 0
            x = x * 1
            x = torch.relu(x)
            return x

    model = TestModel()
    model.eval()

    test_input = torch.randn(32, 10)

    traced = fx.symbolic_trace(model)
    ShapeProp(traced).propagate(test_input)

    graph = fx_to_ir(traced)

    with torch.no_grad():
        naive_output, naive_stats = execute(
            graph,
            model,
            test_input,
            collect_stats=True,
            free_dead_values=False,
        )

        liveness_output, liveness_stats = execute(
            graph,
            model,
            test_input,
            collect_stats=True,
            free_dead_values=True,
        )

    assert torch.allclose(
        naive_output,
        liveness_output,
        rtol=1e-5,
        atol=1e-6,
    )

    assert (
        liveness_stats["peak_retained_bytes"]
        < naive_stats["peak_retained_bytes"]
    )