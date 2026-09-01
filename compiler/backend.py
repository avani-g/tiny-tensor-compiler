import torch

def free_dead_inputs(node, index, values, last_uses):
    for inp in node.inputs:
        if (
            isinstance(inp, str)
            and inp in values
            and last_uses.get(inp) == index
        ):
            del values[inp]

def live_tensor_bytes(values):
    total = 0

    for value in values.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()

    return total

def execute(
    graph,
    model,
    inputs,
    collect_stats=False,
    free_dead_values=True,
):
    values = {}
    last_uses = graph.compute_last_uses()
    peak_retained_bytes = 0

    for index, node in enumerate(graph.nodes):

        if node.op == "input":
            values[node.name] = inputs

        elif node.op == "linear":
            linear_module = getattr(model, node.target)
            input_value = values[node.inputs[0]]
            values[node.name] = linear_module(input_value)

        elif node.op == "relu":
            input_value = values[node.inputs[0]]
            values[node.name] = torch.relu(input_value)

        elif node.op == "add":
            left = get_value(node.inputs[0], values)
            right = get_value(node.inputs[1], values)
            values[node.name] = left + right

        elif node.op == "mul":
            left = get_value(node.inputs[0], values)
            right = get_value(node.inputs[1], values)
            values[node.name] = left * right

        elif node.op == "fused_linear_relu":
            linear_module = getattr(model, node.target)
            input_value = values[node.inputs[0]]
            linear_output = linear_module(input_value)
            values[node.name] = torch.relu(linear_output)

        elif node.op == "return":
            result = get_value(node.inputs[0], values)

            if collect_stats:
                return result, {
                    "peak_retained_bytes": peak_retained_bytes
                }

            return result

        peak_retained_bytes = max(
            peak_retained_bytes,
            live_tensor_bytes(values),
        )

        if free_dead_values:
            free_dead_inputs(
                node,
                index,
                values,
                last_uses,
            )

    raise RuntimeError("Graph did not contain a return node")


def get_value(value, values):
    if isinstance(value, str):
        return values[value]

    return value