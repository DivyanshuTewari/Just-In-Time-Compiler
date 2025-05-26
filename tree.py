from lark import Lark, Transformer

grammar = """
?expr: term
     | expr "+" term   -> add
     | expr "-" term   -> sub

?term: factor
     | term "*" factor -> mul
     | term "/" factor -> div

?factor: NUMBER        -> number
       | "(" expr ")"

%import common.NUMBER
%import common.WS_INLINE
%ignore WS_INLINE
"""

parser = Lark(grammar, parser='lalr')

class ASTBuilder(Transformer):
    def number(self, n):
        return Number(float(n[0]))

    def add(self, items):
        return BinaryOp(items[0], '+', items[1])

    def sub(self, items):
        return BinaryOp(items[0], '-', items[1])

    def mul(self, items):
        return BinaryOp(items[0], '*', items[1])

    def div(self, items):
        return BinaryOp(items[0], '/', items[1])


class Number:
    def __init__(self, value):
        self.value = value

class BinaryOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


























from llvmlite import ir, binding

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="jit_module")
        self.builder = None
        self.func = None

    def compile(self, ast):
        func_type = ir.FunctionType(ir.DoubleType(), [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        result = self.codegen(ast)
        self.builder.ret(result)

    def codegen(self, node):
        if isinstance(node, Number):
            return ir.Constant(ir.DoubleType(), node.value)
        elif isinstance(node, BinaryOp):
            l = self.codegen(node.left)
            r = self.codegen(node.right)
            if node.op == '+': return self.builder.fadd(l, r)
            if node.op == '-': return self.builder.fsub(l, r)
            if node.op == '*': return self.builder.fmul(l, r)
            if node.op == '/': return self.builder.fdiv(l, r)
        raise TypeError("Unknown node")

























def run_jit(module):
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    target = binding.Target.from_default_triple()
    tm = target.create_target_machine()
    backing_mod = binding.parse_assembly(str(module))
    engine = binding.create_mcjit_compiler(backing_mod, tm)
    engine.finalize_object()

    func_ptr = engine.get_function_address("main")
    import ctypes
    return ctypes.CFUNCTYPE(ctypes.c_double)(func_ptr)()
