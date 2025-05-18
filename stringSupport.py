import re
import ply.lex as lex
import ply.yacc as yacc
import struct
import ctypes

# --------------------------
# Lexer (Tokenization)
# --------------------------
tokens = (
    'NUMBER', 'FLOAT', 'IDENTIFIER', 'PLUS', 'MINUS', 'MUL', 'DIV',
    'LPAREN', 'RPAREN', 'EQUALS', 'SEMICOLON'
)

t_PLUS = r'\+'
t_MINUS = r'-'
t_MUL = r'\*'
t_DIV = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_EQUALS = r'='
t_SEMICOLON = r';'
t_ignore = ' \t\n'

def t_FLOAT(t):
    r'\d*\.\d+([eE][+-]?\d+)?'
    t.value = float(t.value)
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# --------------------------
# Parser (AST Generation)
# --------------------------
class Number:
    def __init__(self, value):
        self.value = value
        self.is_float = isinstance(value, float)
    def __str__(self):
        return f"Number({self.value}, {'float' if self.is_float else 'int'})"

class Identifier:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"Identifier({self.name})"

class BinOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
        self.is_float = (isinstance(left, Number) and left.is_float) or \
                        (isinstance(right, Number) and right.is_float) or \
                        (isinstance(left, BinOp) and left.is_float) or \
                        (isinstance(right, BinOp) and right.is_float)
    def __str__(self):
        return f"BinOp({self.left}, {self.op}, {self.right}, {'float' if self.is_float else 'int'})"

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'MUL', 'DIV'),
    ('nonassoc', 'LPAREN', 'RPAREN')
)

def p_program(p):
    '''program : statement
               | statement program'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]

def p_statement(p):
    '''statement : assignment
                 | expression SEMICOLON
                 | expression'''
    p[0] = p[1]

def p_assignment(p):
    'assignment : IDENTIFIER EQUALS expression SEMICOLON'
    p[0] = ('assign', p[1], p[3])

def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression MUL expression
                  | expression DIV expression'''
    p[0] = BinOp(p[1], p[2], p[3])

def p_expression_factor(p):
    'expression : factor'
    p[0] = p[1]

def p_factor_number(p):
    '''factor : NUMBER
              | FLOAT'''
    p[0] = Number(p[1])

def p_factor_identifier(p):
    'factor : IDENTIFIER'
    p[0] = Identifier(p[1])

def p_factor_expr(p):
    'factor : LPAREN expression RPAREN'
    p[0] = p[2]

def p_error(p):
    print(f"Syntax error at '{p.value}'")

parser = yacc.yacc(start='program')

# --------------------------
# Code Generation (AST → x86_64 Assembly)
# --------------------------
def compile_ast(node, context, reg='rax'):
    code = []
    is_float = getattr(node, 'is_float', False)

    if isinstance(node, Number):
        if node.is_float:
            # Convert float to binary representation
            float_bits = struct.pack('d', node.value)
            int_bits = struct.unpack('Q', float_bits)[0]
            code.append(f"mov rax, {int_bits}")
            code.append("movq xmm0, rax")
        else:
            code.append(f"mov {reg}, {node.value}")
    elif isinstance(node, Identifier):
        addr = context['variables'][node.name]
        if is_float:
            code.append(f"movsd xmm0, [rbp - {addr}]")
        else:
            code.append(f"mov {reg}, [rbp - {addr}]")
    elif isinstance(node, BinOp):
        if node.is_float:
            # Evaluate left operand into xmm0
            code += compile_ast(node.left, context, 'rax')
            code.append("movsd [rbp - 8], xmm0")  # Save left result
            # Evaluate right operand into xmm1
            code += compile_ast(node.right, context, 'rax')
            code.append("movsd xmm1, xmm0")
            code.append("movsd xmm0, [rbp - 8]")  # Restore left result
            if node.op == '+':
                code.append("addsd xmm0, xmm1")
            elif node.op == '-':
                code.append("subsd xmm0, xmm1")
            elif node.op == '*':
                code.append("mulsd xmm0, xmm1")
            elif node.op == '/':
                code.append("divsd xmm0, xmm1")
        else:
            # Integer operations
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("mov rcx, rax")
            code.append("pop rax")
            if node.op == '+':
                code.append("add rax, rcx")
            elif node.op == '-':
                code.append("sub rax, rcx")
            elif node.op == '*':
                code.append("imul rax, rcx")
            elif node.op == '/':
                code.append("cqo")
                code.append("idiv rcx")
    elif isinstance(node, tuple) and node[0] == 'assign':
        var_name = node[1]
        code += compile_ast(node[2], context, 'rax')
        addr = context['variables'][var_name]
        if is_float:
            code.append(f"movsd [rbp - {addr}], xmm0")
        else:
            code.append(f"mov [rbp - {addr}], rax")
    return code

# --------------------------
# JIT Execution (Windows-compatible)
# --------------------------
def assemble(assembly):
    opcodes = {
        # register/memory forms
        'mov rax, mem':    b'\x48\x8B\x85',   # mov rax, [rbp - offset]
        'mov rcx, mem':    b'\x48\x8B\x8D',   # mov rcx, [rbp - offset]
        'mov mem, rax':    b'\x48\x89\x85',   # mov [rbp - offset], rax
        'mov rcx, rax':    b'\x48\x89\xC1',   # mov rcx, rax

        # arithmetic
        'add rax, rcx':    b'\x48\x01\xC8',
        'sub rax, rcx':    b'\x48\x29\xC8',
        'imul rax, rcx':   b'\x48\x0F\xAF\xC1',
        'cqo':             b'\x48\x99',
        'idiv rcx':        b'\x48\xF7\xF9',

        # stack/frame
        'push rax':        b'\x50',
        'pop rax':         b'\x58',
        'push rbp':        b'\x55',
        'pop rbp':         b'\x5D',
        'mov rbp, rsp':    b'\x48\x89\xE5',
        'mov rsp, rbp':    b'\x48\x89\xEC',
        'sub rsp, imm':    b'\x48\x81\xEC',

        # return
        'ret':             b'\xC3',

        # floating-point
        'movq xmm0, rax':  b'\x66\x48\x0F\x6E\xC0',  # load rax→xmm0
        'movsd xmm0, mem': b'\xF2\x0F\x10\x85',
        'movsd xmm1, mem': b'\xF2\x0F\x10\x8D',
        'movsd mem, xmm0': b'\xF2\x0F\x11\x85',
        'addsd xmm0, xmm1':b'\xF2\x0F\x58\xC1',
        'subsd xmm0, xmm1':b'\xF2\x0F\x5C\xC1',
        'mulsd xmm0, xmm1':b'\xF2\x0F\x59\xC1',
        'divsd xmm0, xmm1':b'\xF2\x0F\x5E\xC1',
        'movsd xmm1, xmm0':b'\xF2\x0F\x10\xC8',
        'movq rax, xmm0':  b'\x66\x48\x0F\xD6\xC0'   # store xmm0→rax
    }

    machine_code = bytearray()
    for line in assembly:
        line = line.strip()

        # 1) exact matches
        if line in opcodes:
            machine_code += opcodes[line]

        # 2) memory loads/stores with offset
        elif line.startswith("mov rax, [rbp -"):
            off = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['mov rax, mem'] + struct.pack('<i', -off)
        elif line.startswith("mov rcx, [rbp -"):
            off = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['mov rcx, mem'] + struct.pack('<i', -off)
        elif line.startswith("movsd xmm0, [rbp -"):
            off = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['movsd xmm0, mem'] + struct.pack('<i', -off)
        elif line.startswith("movsd xmm1, [rbp -"):
            off = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['movsd xmm1, mem'] + struct.pack('<i', -off)
        elif line.startswith("mov [rbp -"):
            off = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['mov mem, rax'] + struct.pack('<i', -off)
        elif line.startswith("movsd [rbp -"):
            off = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['movsd mem, xmm0'] + struct.pack('<i', -off)

        # 3) sub rsp, <imm32>
        elif line.startswith("sub rsp,"):
            imm32 = int(line.split(',')[1].strip())
            machine_code += opcodes['sub rsp, imm'] + struct.pack('<i', imm32)

        # 4) 64-bit immediates into rax or rcx
        elif line.startswith("mov rax,") and '[' not in line:
            imm64 = int(line.split(',')[1].strip())
            machine_code += b'\x48\xB8' + struct.pack('<Q', imm64)
        elif line.startswith("mov rcx,") and '[' not in line:
            imm64 = int(line.split(',')[1].strip())
            machine_code += b'\x48\xB9' + struct.pack('<Q', imm64)

        else:
            raise ValueError(f"Unknown instruction: {line}")

    return bytes(machine_code)




def jit_compile(expression, variables=None, debug=False):
    lexer.input(expression)
    ast_list = parser.parse(expression)
    if ast_list is None:
        raise ValueError("Parsing failed, no AST generated")
    
    if debug:
        print("AST:")
        for stmt in ast_list:
            print(stmt)

    context = {'variables': variables or {}}
    offset = 16
    for stmt in ast_list:
        if isinstance(stmt, tuple) and stmt[0] == 'assign':
            var = stmt[1]
            if var not in context['variables']:
                context['variables'][var] = offset
                offset += 8

    offset = (offset + 15) & ~15

    assembly = []
    assembly.append("push rbp")
    assembly.append("mov rbp, rsp")
    assembly.append(f"sub rsp, {offset}")
    for stmt in ast_list:
        assembly += compile_ast(stmt, context)
    # Move result to rax for return (for both int and float)
    is_float = any(getattr(stmt, 'is_float', False) for stmt in ast_list if isinstance(stmt, BinOp))
    if is_float:
        assembly.append("movq rax, xmm0")  # Move float result to rax
    assembly.append("mov rsp, rbp")
    assembly.append("pop rbp")
    assembly.append("ret")

    if debug:
        print("\nAssembly:")
        for line in assembly:
            print(line)

    machine_code = assemble(assembly)
    buf = ctypes.create_string_buffer(machine_code)
    old_protect = ctypes.c_ulong()
    ctypes.windll.kernel32.VirtualProtect(
        ctypes.cast(buf, ctypes.c_void_p),
        len(machine_code),
        0x40,  # PAGE_EXECUTE_READWRITE
        ctypes.byref(old_protect)
    )
    func_type = ctypes.CFUNCTYPE(ctypes.c_double if is_float else ctypes.c_int64)
    return func_type(ctypes.addressof(buf))

# --------------------------
# Test Cases
# --------------------------
if __name__ == "__main__":
    # Test 1: Integer operation
    expr = "2 * 3 + 5"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 11

    # Test 2: Floating-point operation
    expr = "3.14 * 2.0 + 1.5"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 7.78

    # Test 3: Mixed integer and float
    expr = "5 + 2.5 * 2"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 10.0

    # Test 4: Variable assignment with floats
    expr = "a=3.14; b=2.0; a * b"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 6.28
    
     # Test 1: Correct precedence (2*3*5 + 6*7 = 30 + 42 = 72)
    expr = "8*3*6*4*7-1-2-4+4+6*24*213*54"
    func = jit_compile(expr)
    result = func()
    print(f"{expr} = {result}")  # Should output: 72

    # Test 2: Parentheses override precedence ((2+3)*5 = 25)
    expr = "(2+3)*5"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 25

    # Test 3: Division and subtraction ((10-2)/4 = 2)
    expr = "(10-2)/4"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 2

    # Test 4: Variable assignment (a=5; b=3; a*b + 2 = 17)
    expr = "a=5; b=3; a*b + 2"
    func = jit_compile(expr)
    print(f"{expr} = {func()}")  # Should output: 17