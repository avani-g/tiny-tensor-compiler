from compiler.ir import IRGraph


def find_linear_relu_fusions(graph: IRGraph):
    users = graph.build_users()
    candidates = []

    for node in graph.nodes:
        if node.op != "linear":
            continue

        node_users = users[node.name]

        if len(node_users) != 1:
            continue

        user_name = node_users[0]

        user_node = graph.get_node(user_name)

        if user_node is None:
            continue

        if user_node.op == "relu":
            candidates.append((node.name, user_node.name))

    return candidates


def fuse_linear_relu(graph: IRGraph) -> bool:
    candidates = find_linear_relu_fusions(graph)

    if not candidates:
        return False

    for linear_name, relu_name in candidates:
        linear_node = graph.get_node(linear_name)
        relu_node = graph.get_node(relu_name)

        if linear_node is None or relu_node is None:
            continue

        fused_name = f"{linear_name}_relu"

        linear_node.name = fused_name
        linear_node.op = "fused_linear_relu"

        for node in graph.nodes:
            node.inputs = [
                fused_name if inp == relu_name else inp
                for inp in node.inputs
            ]

        graph.nodes = [
            node
            for node in graph.nodes
            if node.name != relu_name
        ]

    graph.rebuild_node_map()

    return True