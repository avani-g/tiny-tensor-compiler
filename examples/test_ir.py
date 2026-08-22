from compiler.ir import IRNode, IRGraph

graph = IRGraph()

graph.add_node(IRNode("x", "input", []))
graph.add_node(IRNode("linear", "linear", ['x']))
graph.add_node(IRNode("relu", "relu", ["linear"]))

print(graph)