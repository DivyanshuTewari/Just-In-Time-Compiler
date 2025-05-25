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
