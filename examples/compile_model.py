import torch
import torch.nn as nn
import torch.fx as fx

from compiler.frontend import fx_to_ir
from compiler.passes import (
    PassManager,
    algebraic_simplification,
    constant_folding,
    dead_code_elimination,
)


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

print("FX graph:")
print(traced.graph)

ir_graph = fx_to_ir(traced)

manager = PassManager()
manager.add_pass(algebraic_simplification)
manager.add_pass(constant_folding)
manager.add_pass(dead_code_elimination)

print("\nOriginal IR:")
print(ir_graph)

optimized_graph = manager.run(ir_graph)

print("\nOptimized IR:")
print(optimized_graph)