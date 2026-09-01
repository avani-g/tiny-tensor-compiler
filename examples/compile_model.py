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

from compiler.fusion import find_linear_relu_fusions
from compiler.fusion import fuse_linear_relu

from compiler.backend import execute

from compiler.analysis import analyze_graph


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


model = TinyModel()

traced = fx.symbolic_trace(model)

example_input = torch.randn(32, 10)

ShapeProp(traced).propagate(example_input)

print("FX graph:")
print(traced.graph)

print("\nFX node metadata:")
for node in traced.graph.nodes:
    print(node.name, node.meta.get("tensor_meta"))

ir_graph = fx_to_ir(traced)

before_stats = analyze_graph(ir_graph)

print("\nBefore optimization:")
print(ir_graph)
print(before_stats)

manager = PassManager()

manager.add_pass(algebraic_simplification)
manager.add_pass(constant_folding)
manager.add_pass(dead_code_elimination)
manager.add_pass(fuse_linear_relu)

optimized_graph = manager.run(ir_graph)

print("\nOptimized IR:")
print(optimized_graph)

fusion_candidates = find_linear_relu_fusions(optimized_graph)

test_input = torch.randn(32, 10)

model.eval()

with torch.no_grad():
    pytorch_output = model(test_input)


with torch.no_grad():
    compiled_output = execute(
        optimized_graph,
        model,
        test_input,
    )

print(
    "\nOutputs match:",
    torch.allclose(
        pytorch_output,
        compiled_output,
        rtol=1e-5,
        atol=1e-6,
    )
)

after_stats = analyze_graph(optimized_graph)

print("\nAfter optimization:")
print(optimized_graph)
print(after_stats)

def reduction_percent(before, after):
    if before == 0:
        return 0.0

    return ((before - after) / before) * 100


print("\nOptimization results:")

print(
    "Node reduction:",
    f"{reduction_percent(before_stats['node_count'], after_stats['node_count']):.1f}%"
)

print(
    "Estimated peak memory reduction:",
    f"{reduction_percent(before_stats['peak_memory_bytes'], after_stats['peak_memory_bytes']):.1f}%"
)