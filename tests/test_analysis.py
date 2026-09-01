from compiler.ir import IRNode, IRGraph
from compiler.analysis import analyze_graph, compare_graphs


def test_analyze_graph():
    graph = IRGraph()

    graph.add_node(
        IRNode(
            "x",
            "input",
            [],
            shape=[10],
            dtype="torch.float32",
        )
    )

    graph.add_node(
        IRNode(
            "a",
            "relu",
            ["x"],
            shape=[20],
            dtype="torch.float32",
        )
    )

    graph.add_node(
        IRNode(
            "output",
            "return",
            ["a"],
        )
    )

    stats = analyze_graph(graph)

    assert stats["node_count"] == 3
    assert stats["peak_memory_bytes"] == 120

def test_compare_graphs():
    before = IRGraph()

    before.add_node(
        IRNode(
            "x",
            "input",
            [],
            shape=[10],
            dtype="torch.float32",
        )
    )

    before.add_node(
        IRNode(
            "a",
            "relu",
            ["x"],
            shape=[10],
            dtype="torch.float32",
        )
    )

    before.add_node(
        IRNode(
            "b",
            "relu",
            ["a"],
            shape=[10],
            dtype="torch.float32",
        )
    )

    before.add_node(
        IRNode(
            "output",
            "return",
            ["b"],
        )
    )

    after = IRGraph()

    after.add_node(
        IRNode(
            "x",
            "input",
            [],
            shape=[10],
            dtype="torch.float32",
        )
    )

    after.add_node(
        IRNode(
            "b",
            "relu",
            ["x"],
            shape=[10],
            dtype="torch.float32",
        )
    )

    after.add_node(
        IRNode(
            "output",
            "return",
            ["b"],
        )
    )

    comparison = compare_graphs(before, after)

    assert comparison["before"]["node_count"] == 4
    assert comparison["after"]["node_count"] == 3

    assert comparison["node_reduction_percent"] == 25.0