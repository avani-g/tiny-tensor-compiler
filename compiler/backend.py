import torch


def execute(graph, model, inputs):
    values = {}

    for node in graph.nodes:
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
            return get_value(node.inputs[0], values)

    raise RuntimeError("Graph did not contain a return node")


def get_value(value, values):
    if isinstance(value, str):
        return values[value]

    return value