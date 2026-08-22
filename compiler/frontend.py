import torch
import torch.fx as fx

from compiler.ir import IRNode, IRGraph


def fx_to_ir(traced: fx.GraphModule) -> IRGraph:
    ir_graph = IRGraph()

    for node in traced.graph.nodes:

        if node.op == "placeholder":
            ir_graph.add_node(IRNode(name=node.name, op="input", inputs=[]))

        elif node.op == "call_module":
            inputs = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
            ir_graph.add_node(IRNode(name=node.name, op=str(node.target), inputs=inputs))

        elif node.op == "call_function":
            inputs = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
            ir_graph.add_node(IRNode(name=node.name, op=str(node.target.__name__), inputs=inputs))

        elif node.op == "output":
            inputs = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
            ir_graph.add_node(IRNode(name="output", op="return", inputs=inputs))

    return ir_graph