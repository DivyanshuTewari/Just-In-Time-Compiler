import re
import ply.lex as lex
import ply.yacc as yacc
import struct
import ctypes

# --------------------------
# Lexer (Tokenization)
# --------------------------
tokens = (
    'NUMBER', 'IDENTIFIER', 'STRING',
    'PLUS', 'MINUS', 'MUL', 'DIV',
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

def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    t.value = bytes(t.value[1:-1], 'utf-8').decode('unicode_escape').encode('utf-8') + b'\x00'
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
    def __init__(self, value): self.value = value
class Identifier:
    def __init__(self, name): self.name = name
class String:
    def __init__(self, value): self.value = value
class BinOp:
    def __init__(self, left, op, right): self.left = left; self.op = op; self.right = right

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'MUL', 'DIV'),
    ('nonassoc', 'LPAREN', 'RPAREN')
)

def p_program(p):
    '''program : statement
               | statement program'''
    if len(p) == 2: p[0] = [p[1]]
    else: p[0] = [p[1]] + p[2]

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

def p_factor_string(p):
    'factor : STRING'
    p[0] = String(p[1])

def p_factor_expr(p):
    'factor : LPAREN expression RPAREN'
    p[0] = p[2]

def p_error(p):
    print(f"Syntax error at '{getattr(p, 'value', '?')}'")

parser = yacc.yacc(start='program')

# --------------------------
# String Buffer Management
# --------------------------
STATIC_STRING_BUFFER_SIZE = 4096
_static_string_buffer = ctypes.create_string_buffer(STATIC_STRING_BUFFER_SIZE)
_static_string_next = 0

def allocate_static_string(s):
    global _static_string_next
    n = len(s)
    if _static_string_next + n > STATIC_STRING_BUFFER_SIZE:
        raise RuntimeError("Static string buffer overflow")
    offset = _static_string_next
    ctypes.memmove(ctypes.addressof(_static_string_buffer) + offset, s, n)
    _static_string_next += n
    return ctypes.addressof(_static_string_buffer) + offset

def get_string(addr):
    return ctypes.string_at(addr)

def is_string_node(node, context):
    if isinstance(node, String):
        return True
    if isinstance(node, Identifier):
        return node.name in context['string_vars']
    if isinstance(node, BinOp) and node.op == '+':
        return is_string_node(node.left, context) and is_string_node(node.right, context)
    return False

def eval_string_node(node, context):
    if isinstance(node, String):
        return node.value
    elif isinstance(node, Identifier):
        addr = context['variables'][node.name]
        return ctypes.string_at(addr)
    elif isinstance(node, BinOp) and node.op == '+':
        left = eval_string_node(node.left, context)
        right = eval_string_node(node.right, context)
        if left.endswith(b'\x00'):
            left = left[:-1]
        return left + right
    else:
        raise RuntimeError("Not a string node")

# --------------------------
# Code Generation (AST → x86_64 Assembly)
# --------------------------
def compile_ast(node, context, reg='rax'):
    code = []
    if isinstance(node, Number):
        code.append(f"mov {reg}, {node.value}")
    elif isinstance(node, String):
        addr = context['string_buffer_offsets'][node.value]
        code.append(f"mov {reg}, {addr}")
    elif isinstance(node, Identifier):
        if node.name in context['string_vars']:
            addr = context['variables'][node.name]
            code.append(f"mov {reg}, {addr}")
        elif line == "mov rax, rax":
            machine_code += opcodes['mov rax, rax']
        elif line.startswith("mov rax,") and '[' not in line:
            arg = line.split(',')[1].strip()
            if arg.isdigit() or (arg[0] == '-' and arg[1:].isdigit()):
                imm = int(arg)
                i3647:
                    machine_code += opcodes['mov rax, imm'] + struct.pack('<i', imm)
                else:
                    machine_code += opcodes['mov rax, imm64'] + struct.pack('<Q', imm)
            else:
                raise ValueError(f"Unknown or unsupported mov rax, {arg}")
        elif line.startswith("mov rcx,") and '[' not in line:
            arg = line.split(',')[1].strip()
            if arg.isdigit() or (arg[0] == '-' and arg[1:].isdigit()):
                imm = int(arg)
                machine_code += opcodes['mov rcx, imm'] + struct.pack('<i', imm)
            elif arg == 'rax':
                machine_code += opcodes['mov rcx, rax']
            else:
                raise ValueError(f"Unknown or unsupported mov rcx, {arg}")
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
        elif line.startswith("lea rax, [rbp -"):
            offset = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['lea rax, mem'] + struct.pack('<i', -offset)
        elif line.startswith("lea rcx, [rbp -"):
            offset = int(line.split('-')[1].split(']')[0].strip())
            machine_code += opcodes['lea rcx, mem'] + struct.pack('<i', -offset)
        elif line.startswith("mov byte ptr [rbp -"):
            m = re.match(r"mov byte ptr \[rbp - (\d+)\s*\+\s*(\d+)\],\s*(\d+)", line)
            if m:
                base = int(m.group(1))
                plus = int(m.group(2))
                offset = base + plus
                imm = int(m.group(3))
            else:
                m = re.match(r"mov byte ptr \[rbp - (\d+)\],\s*(\d+)", line)
                if not m:
                    raise ValueError(f"Unknown instruction: {line}")
                offset = int(m.group(1))
                imm = int(m.group(2))
            machine_code += opcodes['mov byte ptr'] + struct.pack('<i', -offset) + bytes([imm])
        else:
            raise ValueError(f"Unknown instruction: {line}")
    return bytes(machine_code)

def jit_compile(expression, variables=None, debug=False):
    global _static_string_next
    lexer.input(expression)
    ast_list = parser.parse(expression)
    if ast_list is None:
        raise ValueError("Parsing failed, no AST generated")

    _static_string_next = 0
    string_buffer_offsets = {}
    var_offsets = {}
    variables = {} if variables is None else variables.copy()
    string_vars = set()
    offset = 16

    def collect_strings(node):
        if isinstance(node, String):
            if node.value not in string_buffer_offsets:
                string_buffer_offsets[node.value] = allocate_static_string(node.value)
        elif isinstance(node, BinOp):
            collect_strings(node.left)
            collect_strings(node.right)
        elif isinstance(node, tuple) and node[0] == 'assign':
            collect_strings(node[2])

    for stmt in ast_list:
        collect_strings(stmt)

    def collect_vars(node):
        nonlocal offset
        if isinstance(node, tuple) and node[0] == 'assign':
            var = node[1]
            val = node[2]
            if not is_string_node(val, {'string_vars': set()}):
                if var not in var_offsets:
                    var_offsets[var] = offset
                    offset += 8
        elif isinstance(node, BinOp):
            collect_vars(node.left)
            collect_vars(node.right)

    for stmt in ast_list:
        collect_vars(stmt)


    if offset % 16 != 0:
        offset += (16 - (offset % 16))

    context = {
        'variables': variables,

