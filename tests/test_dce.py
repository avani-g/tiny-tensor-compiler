from compiler.ir import IRNode, IRGraph
from compiler.passes import dead_code_elimination


def test_dead_branch_removed():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))

    graph.add_node(IRNode("a", "relu", ["x"]))
    graph.add_node(IRNode("dead", "mul", ["x", 10]))

    graph.add_node(IRNode("output", "return", ["a"]))

    changed = dead_code_elimination(graph)

    assert changed is True

    assert graph.get_node("dead") is None
    assert graph.get_node("a") is not None
    assert graph.get_node("x") is not None