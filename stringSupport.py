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
        elif node.name in context['variables']:
            addr = context['variables'][node.name]
            code.append(f"mov {reg}, [rbp - {addr}]")
        else:
            raise ValueError(f"Undefined variable '{node.name}'")
    elif isinstance(node, BinOp):
        left_is_str = is_string_node(node.left, context)
        right_is_str = is_string_node(node.right, context)
        if node.op == '+' and left_is_str and right_is_str:
            left_bytes = eval_string_node(node.left, context)
            right_bytes = eval_string_node(node.right, context)
            if left_bytes.endswith(b'\x00'):
                left_bytes = left_bytes[:-1]
            concat = left_bytes + right_bytes
            # Ensure null-termination
            result_buf = ctypes.create_string_buffer(len(concat) + 1)
            ctypes.memmove(result_buf, concat, len(concat))
            result_buf[len(concat)] = 0
            context['concat_buffers'].append(result_buf)
            code.append(f"mov {reg}, {ctypes.addressof(result_buf)}")
        elif node.op == '+':
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("mov rcx, rax")
            code.append("pop rax")
            code.append("add rax, rcx")
        elif node.op == '-':
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("mov rcx, rax")
            code.append("pop rax")
            code.append("sub rax, rcx")
        elif node.op == '*':
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("mov rcx, rax")
            code.append("pop rax")
            code.append("imul rax, rcx")
        elif node.op == '/':
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("mov rcx, rax")
            code.append("pop rax")
            code.append("cqo")
            code.append("idiv rcx")
        else:
            raise RuntimeError(f"Unsupported operator: {node.op}")
    elif isinstance(node, tuple) and node[0] == 'assign':
        var_name = node[1]
        value = node[2]
        if is_string_node(value, context):
            val_bytes = eval_string_node(value, context)
            addr = allocate_static_string(val_bytes)
            context['variables'][var_name] = addr
            context['string_vars'].add(var_name)
        else:
            code += compile_ast(value, context, 'rax')
            addr = context['var_offsets'][var_name]
            code.append(f"mov [rbp - {addr}], rax")
            context['variables'][var_name] = addr
            if var_name in context['string_vars']:
                context['string_vars'].remove(var_name)
    return code

def assemble(assembly):
    opcodes = {
        'mov rax, imm': b'\x48\xC7\xC0',
        'mov rax, imm64': b'\x48\xb8',
        'mov rcx, imm': b'\x48\xC7\xC1',
        'mov rax, mem': b'\x48\x8B\x85',
        'mov rcx, mem': b'\x48\x8B\x8D',
        'mov mem, rax': b'\x48\x89\x85',
        'add rax, rcx': b'\x48\x01\xC8',
        'sub rax, rcx': b'\x48\x29\xC8',
        'imul rax, rcx': b'\x48\x0F\xAF\xC1',
        'cqo': b'\x48\x99',
        'idiv rcx': b'\x48\xF7\xF9',
        'push rax': b'\x50',
        'pop rax': b'\x58',
        'push rbp': b'\x55',
        'pop rbp': b'\x5D',
        'mov rbp, rsp': b'\x48\x89\xE5',
        'mov rsp, rbp': b'\x48\x89\xEC',
        'lea rax, mem': b'\x48\x8D\x85',
        'lea rcx, mem': b'\x48\x8D\x8D',        'mov byte ptr': b'\xC6\x85',
        'ret': b'\xC3',
        'sub rsp, imm': b'\x48\x81\xec',
        'mov rcx, rax': b'\x48\x89\xc1',
        'mov rax, rax': b'\x48\x89\xc0',
    }

    machine_code = bytearray()
    for line in assembly:
        line = line.strip()
        if line in opcodes:
            machine_code += opcodes[line]
        elif line == "mov rcx, rax":
            machine_code += opcodes['mov rcx, rax']
        elif line == "mov rax, rax":
            machine_code += opcodes['mov rax, rax']
        elif line.startswith("mov rax,") and '[' not in line:
            arg = line.split(',')[1].strip()
            if arg.isdigit() or (arg[0] == '-' and arg[1:].isdigit()):
                imm = int(arg)
                if -2147483648 <= imm <= 2147483647:
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

