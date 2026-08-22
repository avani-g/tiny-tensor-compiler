import torch
import torch.nn as nn
import torch.fx as fx

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

print(traced.graph)

print("Nodes:\n")

for node in traced.graph.nodes:
    print("Name:", node.name)
    print("Operation:", node.op)
    print("Target:", node.target)
    print("Arguments:", node.args)
    print()