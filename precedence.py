import re
import ply.lex as lex
import ply.yacc as yacc
import struct
import ctypes

# --------------------------
# Lexer (Tokenization)
# --------------------------
tokens = (
    'NUMBER', 'IDENTIFIER', 'PLUS', 'MINUS', 'MUL', 'DIV',
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
    def __str__(self):
        return f"Number({self.value})"

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
    def __str__(self):
        return f"BinOp({self.left}, {self.op}, {self.right})"

# Precedence rules
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
    'factor : NUMBER'
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
    if isinstance(node, Number):
        code.append(f"mov {reg}, {node.value}")
    elif isinstance(node, Identifier):
        addr = context['variables'][node.name]
        code.append(f"mov {reg}, [rbp - {addr}]")
    elif isinstance(node, BinOp):
        # Evaluate left operand into rax
        code += compile_ast(node.left, context, 'rax')
        code.append("push rax")  # Save left result

        # Evaluate right operand into rax (reuse register safely)
        code += compile_ast(node.right, context, 'rax')
        code.append("mov rcx, rax")  # Move right result to rcx

        code.append("pop rax")  # Restore left result

        if node.op == '+':
            code.append("add rax, rcx")
        elif node.op == '-':
            code.append("sub rax, rcx")
        elif node.op == '*':
            code.append("imul rax, rcx")
        elif node.op == '/':
            code.append("cqo")  # Sign-extend rax into rdx:rax
            code.append("idiv rcx")  # Divide rdx:rax by rcx, result in rax
    elif isinstance(node, tuple) and node[0] == 'assign':
        var_name = node[1]
        code += compile_ast(node[2], context, 'rax')
        addr = context['variables'][var_name]
        code.append(f"mov [rbp - {addr}], rax")
    return code

# --------------------------
# JIT Execution (Windows-compatible)
# --------------------------
def assemble(assembly):
    opcodes = {
        'mov rax, imm': b'\x48\xC7\xC0',  # mov rax, immediate
        'mov rcx, imm': b'\x48\xC7\xC1',  # mov rcx, immediate
        'mov rax, mem': b'\x48\x8B\x85',  # mov rax, [rbp - offset]
        'mov rcx, mem': b'\x48\x8B\x8D',  # mov rcx, [rbp - offset]
        'mov mem, rax': b'\x48\x89\x85',  # mov [rbp - offset], rax
        'add rax, rcx': b'\x48\x01\xC8',  # add rax, rcx
        'sub rax, rcx': b'\x48\x29\xC8',  # sub rax, rcx
        'imul rax, rcx': b'\x48\x0F\xAF\xC1',  # imul rax, rcx
        'cqo': b'\x48\x99',  # sign-extend rax into rdx:rax
        'idiv rcx': b'\x48\xF7\xF9',  # idiv rcx
        'push rax': b'\x50',  # push rax
        'pop rax': b'\x58',  # pop rax
        'push rbp': b'\x55',  # push rbp
        'pop rbp': b'\x5D',  # pop rbp
        'mov rbp, rsp': b'\x48\x89\xE5',  # mov rbp, rsp
        'mov rsp, rbp': b'\x48\x89\xEC',  # mov rsp, rbp
        'sub rsp, imm': b'\x48\x81\xEC',  # sub rsp, immediate
        'mov rcx, rax': b'\x48\x89\xC1',  # mov rcx, rax
        'ret': b'\xC3'  # ret
    }

    machine_code = bytearray()
    for line in assembly:
        line = line.strip()
        # Check opcodes dictionary first
        if line in opcodes:
            machine_code += opcodes[line]
        elif line.startswith("mov rax,") and '[' not in line:
            imm = int(line.split(',')[1].strip())
            machine_code += opcodes['mov rax, imm'] + struct.pack('<i', imm)
        elif line.startswith("mov rcx,") and '[' not in line:
            imm = int(line.split(',')[1].strip())
            machine_code += opcodes['mov rcx, imm'] + struct.pack('<i', imm)
        elif line.startswith("mov rax, [rbp -"):
            offset = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['mov rax, mem'] + struct.pack('<i', -offset)
        elif line.startswith("mov rcx, [rbp -"):
            offset = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['mov rcx, mem'] + struct.pack('<i', -offset)
        elif line.startswith("mov [rbp -"):
            offset = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['mov mem, rax'] + struct.pack('<i', -offset)
        elif line.startswith("sub rsp,"):
            imm = int(line.split(',')[1].strip())
            machine_code += opcodes['sub rsp, imm'] + struct.pack('<i', imm)
        else:
            raise ValueError(f"Unknown instruction: {line}")
    return bytes(machine_code)

def jit_compile(expression, variables=None, debug=False):
    lexer.input(expression)
    ast_list = parser.parse(expression)
    if ast_list is None:
        raise ValueError("Parsing failed, no AST generated")
    
    # Debug: Print AST
    if debug:
        print("AST:")
        for stmt in ast_list:
            print(stmt)

    context = {'variables': variables or {}}
    offset = 16  # Start with 16 bytes for alignment
    for stmt in ast_list:
        if isinstance(stmt, tuple) and stmt[0] == 'assign':
            var = stmt[1]
            if var not in context['variables']:
                context['variables'][var] = offset
                offset += 8

    # Ensure offset is 16-byte aligned
    offset = (offset + 15) & ~15

    assembly = []
    assembly.append("push rbp")  # Save base pointer (8 bytes)
    assembly.append("mov rbp, rsp")  # Set up stack frame
    assembly.append(f"sub rsp, {offset}")  # Allocate stack space (16-byte aligned)
    for stmt in ast_list:
        assembly += compile_ast(stmt, context)
    assembly.append("mov rsp, rbp")  # Restore stack pointer
    assembly.append("pop rbp")  # Restore base pointer
    assembly.append("ret")

    # Debug: Print assembly
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
    func_type = ctypes.CFUNCTYPE(ctypes.c_int64)  # Use 64-bit return type
    return func_type(ctypes.addressof(buf))

# --------------------------
# Test Cases
# --------------------------
if __name__ == "__main__":
    # Test 1: Correct precedence (2*3*5 + 6*7 = 30 + 42 = 72)
    expr = "8*3*6*4*7-1-2-4+4+6*24*213*54"
    func = jit_compile(expr)
    result = func()
    print(f"{expr} = {result}")  

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