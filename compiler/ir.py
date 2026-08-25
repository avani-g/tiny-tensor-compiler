class IRNode:
    def __init__(self, name, op, inputs, shape=None, dtype=None, target=None):
        self.name = name
        self.op = op
        self.inputs = inputs
        self.shape = shape
        self.dtype = dtype
        self.target = target

    def __repr__(self):
        inputs = ", ".join(str(x) for x in self.inputs)

        metadata = ""

        if self.shape is not None:
            metadata += f" shape={self.shape}"

        if self.dtype is not None:
            metadata += f" dtype={self.dtype}"

        return f"{self.name} = {self.op}({inputs}){metadata}"


class IRGraph:
    def __init__(self):
        self.nodes = []
        self.node_map = {}

    def add_node(self, node):
        self.nodes.append(node)
        self.node_map[node.name] = node

    def get_node(self, name):
        return self.node_map.get(name)

    def rebuild_node_map(self):
        self.node_map = {
            node.name: node
            for node in self.nodes
        }

    def build_users(self):
        users = {node.name: [] for node in self.nodes}

        for node in self.nodes:
            for inp in node.inputs:
                if isinstance(inp, str) and inp in users:
                    users[inp].append(node.name)

        return users

    def __repr__(self):
        return "\n".join(str(node) for node in self.nodes)