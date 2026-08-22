import torch
import torch.nn as nn
import torch.fx as fx

from compiler.frontend import fx_to_ir


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x


model = TinyModel()

traced = fx.symbolic_trace(model)

print("FX graph:")
print(traced.graph)

ir_graph = fx_to_ir(traced)

print("\nOur IR:")
print(ir_graph)