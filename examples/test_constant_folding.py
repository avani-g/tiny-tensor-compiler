from compiler.ir import IRNode, IRGraph
from compiler.passes import constant_folding


graph = IRGraph()

graph.add_node(IRNode("a", "const", [3]))
graph.add_node(IRNode("b", "const", [4]))
graph.add_node(IRNode("c", "add", ["a", "b"]))
graph.add_node(IRNode("d", "const", [2]))
graph.add_node(IRNode("e", "mul", ["c", "d"]))

print("Before:")
print(graph)

constant_folding(graph)

print("\nAfter:")
print(graph)