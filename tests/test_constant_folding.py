from compiler.ir import IRNode, IRGraph
from compiler.passes import constant_folding


def test_constant_addition():
    graph = IRGraph()

    graph.add_node(IRNode("a", "const", [3]))
    graph.add_node(IRNode("b", "const", [4]))
    graph.add_node(IRNode("c", "add", ["a", "b"]))

    changed = constant_folding(graph)

    assert changed is True

    c = graph.get_node("c")

    assert c.op == "const"
    assert c.inputs == [7]


def test_constant_chain():
    graph = IRGraph()

    graph.add_node(IRNode("a", "const", [3]))
    graph.add_node(IRNode("b", "const", [4]))
    graph.add_node(IRNode("c", "add", ["a", "b"]))
    graph.add_node(IRNode("d", "const", [2]))
    graph.add_node(IRNode("e", "mul", ["c", "d"]))

    constant_folding(graph)

    e = graph.get_node("e")

    assert e.op == "const"
    assert e.inputs == [14]