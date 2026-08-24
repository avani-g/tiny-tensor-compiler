from compiler.ir import IRGraph


class PassManager:
    def __init__(self):
        self.passes = []

    def add_pass(self, optimization_pass):
        self.passes.append(optimization_pass)

    def run(self, graph):
        changed = True
        iteration = 0

        while changed:
            iteration += 1
            changed = False

            print(f"\nOptimization iteration {iteration}")

            for optimization_pass in self.passes:
                pass_changed = optimization_pass(graph)

                print(
                    f"{optimization_pass.__name__}: "
                    f"{'changed' if pass_changed else 'no change'}"
                )

                if pass_changed:
                    changed = True

        return graph


def resolve_replacement(value, replacements):
    while value in replacements:
        value = replacements[value]
    return value


def algebraic_simplification(graph: IRGraph) -> bool:
    replacements = {}
    changed = False

    for node in graph.nodes:
        if len(node.inputs) != 2:
            continue

        left = node.inputs[0]
        right = node.inputs[1]

        if node.op == "add" and right == 0:
            replacements[node.name] = left
            changed = True

        elif node.op == "add" and left == 0:
            replacements[node.name] = right
            changed = True

        elif node.op == "mul" and right == 1:
            replacements[node.name] = left
            changed = True

        elif node.op == "mul" and left == 1:
            replacements[node.name] = right
            changed = True

    for node in graph.nodes:
        node.inputs = [
            resolve_replacement(inp, replacements)
            for inp in node.inputs
        ]

    graph.nodes = [
        node
        for node in graph.nodes
        if node.name not in replacements
    ]

    return changed


def constant_folding(graph: IRGraph) -> bool:
    constants = {}
    changed = False

    for node in graph.nodes:
        if node.op == "const":
            constants[node.name] = node.inputs[0]
            continue

        if node.op not in ("add", "mul"):
            continue

        if len(node.inputs) != 2:
            continue

        left = node.inputs[0]
        right = node.inputs[1]

        if left in constants and right in constants:
            left_value = constants[left]
            right_value = constants[right]

            if node.op == "add":
                result = left_value + right_value
            else:
                result = left_value * right_value

            node.op = "const"
            node.inputs = [result]
            constants[node.name] = result

            changed = True

    return changed


def dead_code_elimination(graph: IRGraph) -> bool:
    live = set()

    for node in reversed(graph.nodes):
        if node.op == "return":
            live.add(node.name)

            for inp in node.inputs:
                if isinstance(inp, str):
                    live.add(inp)

        elif node.name in live:
            for inp in node.inputs:
                if isinstance(inp, str):
                    live.add(inp)

    old_size = len(graph.nodes)

    graph.nodes = [
        node
        for node in graph.nodes
        if node.name in live
    ]

    return len(graph.nodes) != old_size