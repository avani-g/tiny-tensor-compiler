class IRNode():
    def __init__(self, name, op, inputs):
        self.name = name
        self.op = op
        self.inputs = inputs 

    def __repr__(self):
        inputs = ", ".join(str(x) for x in self.inputs)
        return f"{self.name} = {self.op}({inputs})"


class IRGraph():
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def __repr__(self):
        return "\n".join(str(node) for node in self.nodes)
