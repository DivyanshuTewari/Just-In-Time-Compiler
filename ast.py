from llvmlite import ir, binding

# Setup LLVM module
module = ir.Module(name="jit_module")
func_type = ir.FunctionType(ir.DoubleType(), [], False)
function = ir.Function(module, func_type, name="main")
block = function.append_basic_block(name="entry")
builder = ir.IRBuilder(block)

class Number:
    def __init__(self, value):
        self.value = value

class BinaryOp:
    def __init__(self, op, left, right):
        self.op = op  # '+', '-', '*', '/'
        self.left = left
        self.right = right


def compile_expr(expr):
    if isinstance(expr, Number):
        return ir.Constant(ir.DoubleType(), expr.value)
    elif isinstance(expr, BinaryOp):
        left_val = compile_expr(expr.left)
        right_val = compile_expr(expr.right)
        if expr.op == '+':
            return builder.fadd(left_val, right_val)
        elif expr.op == '-':
            return builder.fsub(left_val, right_val)
        elif expr.op == '*':
            return builder.fmul(left_val, right_val)
        elif expr.op == '/':
            return builder.fdiv(left_val, right_val)
        else:
            raise ValueError(f"Unknown operator {expr.op}")
    else:
        raise TypeError("Unknown expression type")
