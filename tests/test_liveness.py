from compiler.ir import IRNode, IRGraph


def test_compute_last_uses():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))
    graph.add_node(IRNode("a", "linear", ["x"]))
    graph.add_node(IRNode("b", "relu", ["a"]))
    graph.add_node(IRNode("c", "add", ["a", "b"]))
    graph.add_node(IRNode("output", "return", ["c"]))

    last_uses = graph.compute_last_uses()

    assert last_uses == {
        "x": 1,
        "a": 3,
        "b": 3,
        "c": 4,
    }