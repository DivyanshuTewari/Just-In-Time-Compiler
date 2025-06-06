from lark import Lark, Transformer


from lark import Lark, Transformer

grammar = """
?start: stmt+

?stmt: "let" NAME "=" expr    -> assign
     | expr                   -> expr_stmt

?expr: term
     | expr "+" term          -> add
     | expr "-" term          -> sub

?term: factor
     | term "*" factor        -> mul
     | term "/" factor        -> div

?factor: NUMBER               -> number
       | NAME                 -> var
       | "(" expr ")"

%import common.CNAME -> NAME
%import common.NUMBER
%import common.WS_INLINE
%ignore WS_INLINE
"""

parser = Lark(grammar, parser='lalr')


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






from parser import parser, ASTBuilder

def main():
    print("Mini JIT REPL. Type `exit` to quit.")
    source = ""

    while True:
        try:
            line = input(">>> ")
            if line.strip() == "exit": break
            source += line + "\n"
            tree = parser.parse(source)
            ast = ASTBuilder().transform(tree)

            cg = CodeGen()
            cg.compile(ast)
            print("LLVM IR:")
            print(cg.module)
            result = run_jit(cg.module)
            print("Result:", result)

            source = ""  # Reset after successful execution
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()






class Number:
    def __init__(self, value): self.value = value

class BinaryOp:
    def __init__(self, left, op, right): self.left = left; self.op = op; self.right = right

class Variable:
    def __init__(self, name): self.name = name

class Assignment:
    def __init__(self, name, value): self.name = name; self.value = value

class ExpressionStmt:
    def __init__(self, expr): self.expr = expr

class ASTBuilder(Transformer):
    def number(self, n): return Number(float(n[0]))
    def var(self, name): return Variable(str(name[0]))
   
    def assign(self, items): return Assignment(str(items[0]), items[1])
    def expr_stmt(self, items): return ExpressionStmt(items[0])



class CodeGen:
    def __init__(self):
        
        self.func = None
        self.variables = {}

    def compile(self, stmts):
        func_type = ir.FunctionType(ir.DoubleType(), [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

   

        

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

            
        else:
            raise TypeError(f"Unknown AST node: {type(node)}")












from llvmlite import ir, binding

class CodeGen:
    def __init__(self):
        self.module = ir.Module(name="jit_module")
        self.builder = None
        self.func = None

    def compile(self, ast):
        func_type = ir.FunctionType(ir.DoubleType(), [])
        self.func = ir.Function(self.module, func_type, name="main")
      
        result = self.codegen(ast)
        self.builder.ret(result)

    def codegen(self, node):
        if isinstance(node, Number):
            return ir.Constant(ir.DoubleType(), node.value)
        elif isinstance(node, BinaryOp):
            l = self.codegen(node.left)
            r = self.codegen(node.right)
           
            if node.op == '-': return self.builder.fsub(l, r)
            if node.op == '*': return self.builder.fmul(l, r)
            if node.op == '/': return self.builder.fdiv(l, r)
        raise TypeError("Unknown node")

























def run_jit(module):
    binding.initialize()
    binding.initialize_native_target()

    target = binding.Target.from_default_triple()
    tm = target.create_target_machine()
    engine.finalize_object()

    func_ptr = engine.get_function_address("main")
    import ctypes
    return ctypes.CFUNCTYPE(ctypes.c_double)(func_ptr)()



from parser import parser, ASTBuilder

expr = "(3 + 5) * 2"
tree = parser.parse(expr)
ast = ASTBuilder().transform(tree)

cg = CodeGen()
cg.compile(ast)
print("LLVM IR:")
print(cg.module)

