import torch
import torch.fx as fx

from compiler.ir import IRNode, IRGraph

def convert_arg(arg):
    if isinstance(arg, fx.Node):
        return arg.name
    return arg

def fx_to_ir(traced: fx.GraphModule) -> IRGraph:
    ir_graph = IRGraph()

    for node in traced.graph.nodes:

        if node.op == "placeholder":
            ir_graph.add_node(IRNode(name=node.name, op="input", inputs=[]))

        elif node.op == "call_module":
            inputs = [convert_arg(arg) for arg in node.args]
            ir_graph.add_node(IRNode(name=node.name, op=str(node.target), inputs=inputs))

        elif node.op == "call_function":
            inputs = [convert_arg(arg) for arg in node.args]
            ir_graph.add_node(IRNode(name=node.name, op=str(node.target.__name__), inputs=inputs))

        elif node.op == "output":
            inputs = [convert_arg(arg) for arg in node.args]
            ir_graph.add_node(IRNode(name="output", op="return", inputs=inputs))

    return ir_graph