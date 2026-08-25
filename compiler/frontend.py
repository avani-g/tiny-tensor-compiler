import torch.fx as fx

from compiler.ir import IRNode, IRGraph


def convert_arg(arg):
    if isinstance(arg, fx.Node):
        return arg.name

    return arg


def get_tensor_metadata(node):
    tensor_meta = node.meta.get("tensor_meta")

    if tensor_meta is None:
        return None, None

    shape = list(tensor_meta.shape)
    dtype = str(tensor_meta.dtype)

    return shape, dtype


def fx_to_ir(traced: fx.GraphModule) -> IRGraph:
    ir_graph = IRGraph()

    for node in traced.graph.nodes:
        shape, dtype = get_tensor_metadata(node)

        if node.op == "placeholder":
            ir_graph.add_node(
                IRNode(
                    name=node.name,
                    op="input",
                    inputs=[],
                    shape=shape,
                    dtype=dtype,
                )
            )

        elif node.op == "call_module":
            inputs = [convert_arg(arg) for arg in node.args]

            ir_graph.add_node(
                IRNode(
                    name=node.name,
                    op=str(node.target),
                    inputs=inputs,
                    shape=shape,
                    dtype=dtype,
                    target=str(node.target),
                )
            )

        elif node.op == "call_function":
            inputs = [convert_arg(arg) for arg in node.args]

            ir_graph.add_node(
                IRNode(
                    name=node.name,
                    op=node.target.__name__,
                    inputs=inputs,
                    shape=shape,
                    dtype=dtype,
                )
            )

        elif node.op == "output":
            inputs = [convert_arg(arg) for arg in node.args]

            ir_graph.add_node(
                IRNode(
                    name="output",
                    op="return",
                    inputs=inputs,
                )
            )

    return ir_graph