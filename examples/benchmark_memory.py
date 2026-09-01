import torch
import torch.nn as nn
import torch.fx as fx

from torch.fx.passes.shape_prop import ShapeProp

from compiler.frontend import fx_to_ir
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


model = TinyModel()
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


with torch.no_grad():
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

naive_peak = naive_stats["peak_retained_bytes"]
liveness_peak = liveness_stats["peak_retained_bytes"]

reduction = (
    (naive_peak - liveness_peak)
    / naive_peak
    * 100
)

print("Naive peak retained bytes:", naive_peak)
print("Liveness-aware peak retained bytes:", liveness_peak)
print(f"Reduction: {reduction:.1f}%")