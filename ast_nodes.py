class Number:
    def __init__(self, value): self.value = value
class Bool:
    def __init__(self, value): self.value = value
class Identifier:
    def __init__(self, name): self.name = name
class String:
    def __init__(self, value): self.value = value
class BinOp:
    def __init__(self, left, op, right): 
        self.left = left
        self.op = op 
        self.right = right
class UnOp:
    def __init__(self, op, val): 
        self.op = op
        self.val = val
