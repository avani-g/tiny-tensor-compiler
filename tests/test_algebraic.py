from compiler.ir import IRNode, IRGraph
from compiler.passes import algebraic_simplification


def test_add_zero_removed():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))
    graph.add_node(IRNode("add", "add", ["x", 0]))
    graph.add_node(IRNode("output", "return", ["add"]))

    changed = algebraic_simplification(graph)

    assert changed is True
    assert graph.get_node("add") is None
    assert graph.get_node("output").inputs == ["x"]


def test_mul_one_removed():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))
    graph.add_node(IRNode("mul", "mul", ["x", 1]))
    graph.add_node(IRNode("output", "return", ["mul"]))

    changed = algebraic_simplification(graph)

    assert changed is True
    assert graph.get_node("mul") is None
    assert graph.get_node("output").inputs == ["x"]