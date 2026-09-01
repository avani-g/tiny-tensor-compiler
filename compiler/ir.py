class IRNode:
    def __init__(self, name, op, inputs, shape=None, dtype=None, target=None):
        self.name = name
        self.op = op
        self.inputs = inputs
        self.shape = shape
        self.dtype = dtype
        self.target = target

    def num_elements(self):
        if self.shape is None:
            return 0

        total = 1

        for dimension in self.shape:
            total *= dimension

        return total

    def size_bytes(self):
        bytes_per_element = {
            "torch.float32": 4,
            "torch.float64": 8,
            "torch.float16": 2,
            "torch.bfloat16": 2,
            "torch.int64": 8,
            "torch.int32": 4,
            "torch.bool": 1,
        }

        if self.dtype is None:
            return 0

        element_size = bytes_per_element.get(self.dtype)

        if element_size is None:
            return 0

        return self.num_elements() * element_size

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

    def compute_last_uses(self):
        last_uses = {}

        for index, node in enumerate(self.nodes):
            for inp in node.inputs:
                if isinstance(inp, str) and inp in self.node_map:
                    last_uses[inp] = index

        return last_uses

    def compute_peak_memory(self):
        last_uses = self.compute_last_uses()

        live_memory = 0
        peak_memory = 0

        for index, node in enumerate(self.nodes):
            live_memory += node.size_bytes()

            peak_memory = max(peak_memory, live_memory)

            for inp in node.inputs:
                if (
                    isinstance(inp, str)
                    and inp in self.node_map
                    and last_uses.get(inp) == index
                ):
                    input_node = self.get_node(inp)
                    live_memory -= input_node.size_bytes()

        return peak_memory

    def __repr__(self):
        return "\n".join(str(node) for node in self.nodes)