from compiler.ir import IRNode, IRGraph
from compiler.passes import dead_code_elimination


graph = IRGraph()

graph.add_node(IRNode("x", "input", []))

graph.add_node(IRNode("a", "linear", ["x"]))
graph.add_node(IRNode("b", "relu", ["a"]))
graph.add_node(IRNode("c", "mul", ["b", 2]))

graph.add_node(IRNode("d", "add", ["x", 5]))
graph.add_node(IRNode("e", "mul", ["d", 10]))

graph.add_node(IRNode("output", "return", ["c"]))


print("Before:")
print(graph)

dead_code_elimination(graph)

print("\nAfter:")
print(graph)