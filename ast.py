from llvmlite import ir, binding
from ctypes import CFUNCTYPE, c_double


# Setup LLVM module
module = ir.Module(name="jit_module")
func_type = ir.FunctionType(ir.DoubleType(), [], False)
function = ir.Function(module, func_type, name="main")
block = function.append_basic_block(name="entry")
builder = ir.IRBuilder(block)


# AST for: (4.0 + 2.0) * 3.0
ast = BinaryOp(
    BinaryOp(Number(4.0), '+', Number(2.0)),
    '*',
    Number(3.0)
)
def run_jit(module):
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = binding.parse_assembly(str(module))
    engine = binding.create_mcjit_compiler(backing_mod, target_machine)

    engine.finalize_object()
    func_ptr = engine.get_function_address("main")



from llvmlite import ir, binding

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="jit_module")
        self.builder = None
        self.func = None
        self.variables = {}

    def compile(self, stmts):
        func_type = ir.FunctionType(ir.DoubleType(), [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        result = None
        for stmt in stmts:
            result = self.codegen(stmt)

        self.builder.ret(result if result else ir.Constant(ir.DoubleType(), 0.0))

    def codegen(self, node):
        if isinstance(node, Number):
            return ir.Constant(ir.DoubleType(), node.value)
        elif isinstance(node, Variable):
            if node.name not in self.variables:
                raise NameError(f"Undefined variable '{node.name}'")
            return self.builder.load(self.variables[node.name], name=node.name)
        elif isinstance(node, Assignment):
            val = self.codegen(node.value)
            ptr = self.builder.alloca(ir.DoubleType(), name=node.name)
            self.builder.store(val, ptr)
            self.variables[node.name] = ptr
        elif isinstance(node, ExpressionStmt):
            return self.codegen(node.expr)
        elif isinstance(node, BinaryOp):
            l = self.codegen(node.left)
            r = self.codegen(node.right)
            ops = {
                '+': self.builder.fadd,
                '-': self.builder.fsub,
                '*': self.builder.fmul,
                '/': self.builder.fdiv
            }
            return ops[node.op](l, r)
        else:
            raise TypeError(f"Unknown AST node: {type(node)}")
