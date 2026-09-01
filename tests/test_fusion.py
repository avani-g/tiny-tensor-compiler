from compiler.ir import IRNode, IRGraph
from compiler.fusion import (
    find_linear_relu_fusions,
    fuse_linear_relu,
)


def test_linear_relu_detected():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))

    graph.add_node(
        IRNode(
            "linear",
            "linear",
            ["x"],
            target="linear",
        )
    )

    graph.add_node(IRNode("relu", "relu", ["linear"]))
    graph.add_node(IRNode("output", "return", ["relu"]))

    candidates = find_linear_relu_fusions(graph)

    assert candidates == [("linear", "relu")]


def test_linear_relu_fused():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))

    graph.add_node(
        IRNode(
            "linear",
            "linear",
            ["x"],
            target="linear",
        )
    )

    graph.add_node(IRNode("relu", "relu", ["linear"]))
    graph.add_node(IRNode("output", "return", ["relu"]))

    changed = fuse_linear_relu(graph)

    assert changed is True

    fused = graph.get_node("linear_relu")

    assert fused is not None
    assert fused.op == "fused_linear_relu"
    assert fused.target == "linear"

    assert graph.get_node("relu") is None
    assert graph.get_node("output").inputs == ["linear_relu"]


def test_linear_relu_not_fused_with_multiple_users():
    graph = IRGraph()

    graph.add_node(IRNode("x", "input", []))

    graph.add_node(
        IRNode(
            "linear",
            "linear",
            ["x"],
            target="linear",
        )
    )

    graph.add_node(IRNode("relu", "relu", ["linear"]))
    graph.add_node(IRNode("mul", "mul", ["linear", 2]))
    graph.add_node(IRNode("output", "return", ["relu"]))

    candidates = find_linear_relu_fusions(graph)

    assert candidates == []