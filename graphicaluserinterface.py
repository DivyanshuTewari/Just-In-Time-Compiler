import re
import ply.lex as lex
import ply.yacc as yacc
import struct
import ctypes
from ctypes import string_at

# --------------------------
# Lexer (Tokenization)
# --------------------------+
tokens = (
    'NUMBER', 'IDENTIFIER', 'STRING', 'BOOL',
    'PLUS', 'MINUS', 'MUL', 'DIV',
    'AND', 'OR', 'NOT',
    'EQ', 'NE', 'LT', 'GT', 'LE', 'GE',
    'IF', 'ELSE',
    'LPAREN', 'RPAREN', 'EQUALS', 'SEMICOLON'
)

# Reserved keywords mapping
reserved = {
    'if': 'IF',
    'else': 'ELSE',
    'true': 'BOOL',
    'false': 'BOOL',
    'TRUE': 'BOOL',
    'FALSE': 'BOOL',
}

t_PLUS = r'\+'
t_MINUS = r'-'
t_MUL = r'\*'
t_DIV = r'/'
t_AND = r'&&'
t_OR = r'\|\|'
t_NOT = r'!'
t_EQ = r'=='
t_NE = r'!='
t_LE = r'<='
t_GE = r'>='
t_LT = r'<'
t_GT = r'>'
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
    t.type = reserved.get(t.value, 'IDENTIFIER')
    if t.type == 'BOOL':
        t.value = t.value.lower() == 'true'
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# --------------------------
# Parser (AST Generation)
# --------------------------
class Number:
    def _init_(self, value): self.value = value
class Bool:
    def _init_(self, value): self.value = value
class Identifier:
    def _init_(self, name): self.name = name
class String:
    def _init_(self, value): self.value = value
class BinOp:
    def _init_(self, left, op, right): self.left = left; self.op = op; self.right = right
class UnOp:
    def _init_(self, op, val): self.op = op; self.val = val
class IfElse:
    def _init_(self, cond, then_expr, else_expr):
        self.cond = cond
        self.then_expr = then_expr
        self.else_expr = else_expr

precedence = (
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'EQ', 'NE'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'MUL', 'DIV'),
    ('right', 'NOT'),
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

def p_expression_ifelse(p):
    'expression : IF LPAREN expression RPAREN expression ELSE expression'
    p[0] = IfElse(p[3], p[5], p[7])

def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression MUL expression
                  | expression DIV expression
                  | expression AND expression
                  | expression OR expression
                  | expression EQ expression
                  | expression NE expression
                  | expression LT expression
                  | expression GT expression
                  | expression LE expression
                  | expression GE expression'''
    p[0] = BinOp(p[1], p[2], p[3])

def p_expression_unop(p):
    'expression : NOT expression'
    p[0] = UnOp(p[1], p[2])

def p_expression_factor(p):
    'expression : factor'
    p[0] = p[1]

def p_factor_bool(p):
    'factor : BOOL'
    p[0] = Bool(p[1])

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
# String Buffer Management (unchanged)
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
    if isinstance(node, IfElse):
        return is_string_node(node.then_expr, context) and is_string_node(node.else_expr, context)
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
    elif isinstance(node, IfElse):
        cond = eval_numeric_node(node.cond, context)
        return eval_string_node(node.then_expr if cond else node.else_expr, context)
    else:
        raise RuntimeError("Not a string node")

def eval_numeric_node(node, context):
    # Used for if-else string/number selection
    if isinstance(node, Number):
        return node.value
    elif isinstance(node, Bool):
        return int(node.value)
    elif isinstance(node, Identifier):
        addr = context['variables'][node.name]
        return ctypes.c_int64.from_address(addr).value
    elif isinstance(node, BinOp):
        # Only for simple numeric/boolean cases
        left = eval_numeric_node(node.left, context)
        right = eval_numeric_node(node.right, context)
        if node.op == '+': return left + right
        if node.op == '-': return left - right
        if node.op == '*': return left * right
        if node.op == '/': return left // right
        if node.op in ('&&', 'AND'): return int(bool(left) and bool(right))
        if node.op in ('||', 'OR'): return int(bool(left) or bool(right))
        if node.op in ('==', 'EQ'): return int(left == right)
        if node.op in ('!=', 'NE'): return int(left != right)
        if node.op in ('<', 'LT'): return int(left < right)
        if node.op in ('>', 'GT'): return int(left > right)
        if node.op in ('<=', 'LE'): return int(left <= right)
        if node.op in ('>=', 'GE'): return int(left >= right)
    elif isinstance(node, UnOp):
        if node.op == '!':
            return int(not eval_numeric_node(node.val, context))
    elif isinstance(node, IfElse):
        cond = eval_numeric_node(node.cond, context)
        return eval_numeric_node(node.then_expr if cond else node.else_expr, context)
    raise RuntimeError("Not a numeric node")

# --------------------------
# Code Generation (AST → x86_64 Assembly)
# --------------------------
_label_counter = [0]
def new_label():
    _label_counter[0] += 1
    return f"L{_label_counter[0]}"

def compile_ast(node, context, reg='rax'):
    code = []
    if isinstance(node, Number):
        code.append(f"mov {reg}, {node.value}")
    elif isinstance(node, Bool):
        code.append(f"mov {reg}, {1 if node.value else 0}")
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
        elif node.op in ('&&', 'AND'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("and rax, rcx")
            code.append("setne al")
            code.append("movzx rax, al")
        elif node.op in ('||', 'OR'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("or rax, rcx")
            code.append("setne al")
            code.append("movzx rax, al")
        elif node.op in ('==', 'EQ'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("sete al")
            code.append("movzx rax, al")
        elif node.op in ('!=', 'NE'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setne al")
            code.append("movzx rax, al")
        elif node.op in ('<', 'LT'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setg al")  # SWAPPED!
            code.append("movzx rax, al")
        elif node.op in ('>', 'GT'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setl al")  # SWAPPED!
            code.append("movzx rax, al")
        elif node.op in ('<=', 'LE'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setge al")  # SWAPPED!
            code.append("movzx rax, al")
        elif node.op in ('>=', 'GE'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setle al")  # SWAPPED!
            code.append("movzx rax, al")
        else:
            raise RuntimeError(f"Unsupported operator: {node.op}")
    elif isinstance(node, UnOp):
        if node.op == '!':
            code += compile_ast(node.val, context, 'rax')
            code.append("test rax, rax")
            code.append("sete al")
            code.append("movzx rax, al")
        else:
            raise RuntimeError(f"Unsupported unary operator: {node.op}")
    elif isinstance(node, IfElse):
        # if (cond) then_expr else else_expr
        else_label = new_label()
        end_label = new_label()
        code += compile_ast(node.cond, context, 'rax')
        code.append("test rax, rax")
        code.append(f"jz {else_label}")
        code += compile_ast(node.then_expr, context, reg)
        code.append(f"jmp {end_label}")
        code.append(f"{else_label}:")
        code += compile_ast(node.else_expr, context, reg)
        code.append(f"{end_label}:")
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
            if var_name not in context['variables']:
                context['stack_offset'] += 8
                context['variables'][var_name] = context['stack_offset']
            offset = context['variables'][var_name]
            code.append(f"mov [rbp - {offset}], rax")

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
        'lea rcx, mem': b'\x48\x8D\x8D',
        'mov byte ptr': b'\xC6\x85',
        'ret': b'\xC3',
        'sub rsp, imm': b'\x48\x81\xec',
        'mov rcx, rax': b'\x48\x89\xc1',
        'mov rax, rax': b'\x48\x89\xc0',
        'cmp rcx, rax': b'\x48\x39\xc8',
        'sete al': b'\x0f\x94\xc0',
        'setne al': b'\x0f\x95\xc0',
        'setl al': b'\x0f\x9c\xc0',
        'setg al': b'\x0f\x9f\xc0',
        'setle al': b'\x0f\x9e\xc0',
        'setge al': b'\x0f\x9d\xc0',
        'movzx rax, al': b'\x48\x0f\xb6\xc0',
        'test rax, rax': b'\x48\x85\xc0',
        'and rax, rcx': b'\x48\x21\xc8',
        'or rax, rcx': b'\x48\x09\xc8',
        'pop rcx': b'\x59',
    }
    machine_code = bytearray()
    labels = {}
    unresolved_jumps = []
    for idx, line in enumerate(assembly):
        if line.endswith(':'):
            labels[line[:-1]] = len(machine_code)
            continue
        if line.startswith('jz '):
            label = line.split()[1]
            machine_code += b'\x0f\x84\x00\x00\x00\x00'
            unresolved_jumps.append((len(machine_code)-4, label, 'jz'))
        elif line.startswith('jmp '):
            label = line.split()[1]
            machine_code += b'\xe9\x00\x00\x00\x00'
            unresolved_jumps.append((len(machine_code)-4, label, 'jmp'))
        else:
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
    for pos, label, jtype in unresolved_jumps:
        target = labels[label]
        rel = target - (pos + 4)
        machine_code[pos:pos+4] = struct.pack('<i', rel)
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
        elif isinstance(node, IfElse):
            collect_strings(node.cond)
            collect_strings(node.then_expr)
            collect_strings(node.else_expr)
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
        elif isinstance(node, IfElse):
            collect_vars(node.cond)
            collect_vars(node.then_expr)
            collect_vars(node.else_expr)
        elif isinstance(node, UnOp):
            collect_vars(node.val)

    for stmt in ast_list:
        collect_vars(stmt)

    if offset % 16 != 0:
        offset += (16 - (offset % 16))

    context = {
        'variables': variables,
        'var_offsets': var_offsets,
        'string_vars': string_vars,
        'string_buffer_offsets': string_buffer_offsets,
        'concat_buffers': []
    }

    for k, v in var_offsets.items():
        if k not in context['variables']:
            context['variables'][k] = v

    assembly = []
    assembly.append("push rbp")
    assembly.append("mov rbp, rsp")
    assembly.append(f"sub rsp, {offset}")

    for stmt in ast_list:
        assembly += compile_ast(stmt, context)

    last = ast_list[-1]
    if isinstance(last, (Number, BinOp, Bool, UnOp, IfElse)):
        assembly += compile_ast(last, context, 'rax')
    elif isinstance(last, String):
        addr = context['string_buffer_offsets'][last.value]
        assembly.append(f"mov rax, {addr}")
    elif isinstance(last, Identifier):
        var = last.name
        if var in context['string_vars']:
            addr = context['variables'][var]
            assembly.append(f"mov rax, {addr}")
        else:
            addr = context['variables'][var]
            assembly.append(f"mov rax, [rbp - {addr}]")
    elif isinstance(last, tuple) and last[0] == 'assign':
        var = last[1]
        if var in context['string_vars']:
            addr = context['variables'][var]
            assembly.append(f"mov rax, {addr}")
        else:
            addr = context['variables'][var]
            assembly.append(f"mov rax, [rbp - {addr}]")

    assembly.append("mov rsp, rbp")
    assembly.append("pop rbp")
    assembly.append("ret")

    machine_code = assemble(assembly)
    buf = ctypes.create_string_buffer(machine_code)
    old_protect = ctypes.c_ulong()
    ctypes.windll.kernel32.VirtualProtect(
        ctypes.cast(buf, ctypes.c_void_p),
        len(machine_code),
        0x40,
        ctypes.byref(old_protect)
    )
    func_type = ctypes.CFUNCTYPE(ctypes.c_uint64)
    return func_type(ctypes.addressof(buf))


import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk

# --------------------------
# GUI Implementation
# --------------------------
class JITCompilerGUI:
    def _init_(self, root):
        self.root = root
        self.root.title("JIT Compiler")
        self.root.geometry("900x700")
        
        # Configure the main grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.create_widgets()
        self.setup_layout()
        
        # Store compiled functions to prevent garbage collection
        self.compiled_functions = []
        
    def create_widgets(self):
        # Font settings
        default_font = ('Consolas', 10)
        
        # Input area
        self.input_label = tk.Label(self.root, text="Enter Expression :", font=('Arial', 11, 'bold'))
        self.input_text = scrolledtext.ScrolledText(
            self.root, 
            height=10, 
            wrap=tk.WORD,
            font=default_font,
            padx=5,
            pady=5
        )
        self.input_text.insert(tk.END, '"Hello, " + "World!"')
        
        
        self.vars_text = scrolledtext.ScrolledText(
            self.root, 
            height=5, 
            wrap=tk.WORD,
            font=default_font,
            padx=5,
            pady=5
        )
        
        # Output area
        self.output_label = tk.Label(self.root, text="Result:", font=('Arial', 11, 'bold'))
        self.output_text = scrolledtext.ScrolledText(
            self.root, 
            height=10, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            font=default_font,
            padx=5,
            pady=5
        )
        
        # Buttons
        self.button_frame = tk.Frame(self.root)
        self.compile_btn = ttk.Button(
            self.button_frame, 
            text="Compile & Run", 
            command=self.compile_and_run,
            style='Accent.TButton'
        )
        self.clear_btn = ttk.Button(
            self.button_frame, 
            text="Clear Output", 
            command=self.clear_output
        )
        self.example_btn = ttk.Button(
            self.button_frame, 
            text="Load Example", 
            command=self.show_examples
        )
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=('Arial', 9)
        )
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        
    def setup_layout(self):
        # Input area
        self.input_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=(10, 0))
        self.input_text.grid(row=1, column=0, sticky=tk.NSEW, padx=10, pady=(0, 5))
        

        
        # Output area
        self.output_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=(5, 0))
        self.output_text.grid(row=5, column=0, sticky=tk.NSEW, padx=10, pady=(0, 10))
        
        # Button frame
        self.button_frame.grid(row=6, column=0, sticky=tk.EW, padx=10, pady=(0, 10))
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        
        # Buttons inside button frame
        self.compile_btn.grid(row=0, column=0, sticky=tk.EW, padx=5)
        self.clear_btn.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.example_btn.grid(row=0, column=2, sticky=tk.EW, padx=5)
        
        # Status bar
        self.status_bar.grid(row=7, column=0, sticky=tk.EW)
        
    def compile_and_run(self):
        expression = self.input_text.get("1.0", tk.END).strip()
        if not expression:
            messagebox.showwarning("Warning", "Please enter an expression to compile")
            return
            
        try:
            # Parse variable definitions
            variables = {}
            var_lines = self.vars_text.get("1.0", tk.END).strip().split('\n')
            for line in var_lines:
                if line and '=' in line:
                    var_name, var_value = line.split('=', 1)
                    var_name = var_name.strip()
                    try:
                        # Try to evaluate as number first
                        variables[var_name] = int(var_value.strip())
                    except ValueError:
                        try:
                            # Then try as boolean
                            val = var_value.strip().lower()
                            if val == 'true':
                                variables[var_name] = True
                            elif val == 'false':
                                variables[var_name] = False
                            else:
                                # Treat as string
                                if val.startswith('"') and val.endswith('"'):
                                    val = val[1:-1]
                                variables[var_name] = val
                        except:
                            pass
            
            self.status_var.set("Compiling...")
            self.root.update()
            
            # Compile and run
            func = jit_compile(expression, variables)
            self.compiled_functions.append(func)  # Keep reference to prevent GC
            result = func()
            
            # Determine if result is a string (pointer) or numeric
            try:
                # Try to interpret as string pointer
                string_result = string_at(result).decode('utf-8', errors='replace')
                output = f"String result :\n\n{string_result}"
            except:
                # Treat as numeric result
                output = f"Numeric result:\n\n{result}"
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, output)
            self.output_text.config(state=tk.DISABLED)
            
            self.status_var.set("Compilation successful")
            
        except Exception as e:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Error:\n\n{str(e)}")
            self.output_text.config(state=tk.DISABLED)
            self.status_var.set(f"Error: {str(e)}")
            
    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_var.set("Ready")
        
    def show_examples(self):
        examples = [
            ("String Concatenation", '"Hello, " + "World!"', ''),
            ("Numeric Expression", "(2+3)*5", ''),
            ("Boolean Comparison", "1 < 2", ''),
            ("If-Else Expression", 'if (5 > 3) "big" else "small"', ''),
            ("With Variables", 'a=5;b=10;a+b',''),
            ("String Variable", 's="Python" ; s+" rocks!"', ''),
            ("Complex Math", "8*3*6*4*7-1-2-4+4+6*24*213*54", ''),
            ("Boolean Logic", "a=true;b=false;a && b", '')
        ]
        
        example_window = tk.Toplevel(self.root)
        example_window.title("Example Expressions")
        example_window.geometry("700x500")
        
        # Configure grid
        example_window.grid_rowconfigure(0, weight=1)
        example_window.grid_columnconfigure(0, weight=1)
        
        # Create treeview with scrollbar
        tree_frame = tk.Frame(example_window)
        tree_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Description", "Expression", "Variables"),
            show="headings",
            yscrollcommand=tree_scroll.set,
            selectmode='browse'
        )
        tree_scroll.config(command=self.tree.yview)
        
        self.tree.pack(expand=True, fill=tk.BOTH)
        
        # Configure columns
        self.tree.heading("Description", text="Description")
        self.tree.heading("Expression", text="Expression")
        self.tree.heading("Variables", text="Variables")
        
        self.tree.column("Description", width=200, anchor=tk.W)
        self.tree.column("Expression", width=300, anchor=tk.W)
        self.tree.column("Variables", width=150, anchor=tk.W)
        
        # Add examples
        for desc, expr, vars in examples:
            self.tree.insert("", tk.END, values=(desc, expr, vars))
        
        # Button frame
        button_frame = tk.Frame(example_window)
        button_frame.grid(row=1, column=0, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Use Selected", 
            command=lambda: self.insert_example(example_window)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Close", 
            command=example_window.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
    def insert_example(self, window):
        selected = self.tree.focus()
        if selected:
            item = self.tree.item(selected)
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, item['values'][1])
            self.vars_text.delete("1.0", tk.END)
            self.vars_text.insert(tk.END, item['values'][2])
            window.destroy()

# --------------------------
# Main Application
# --------------------------
if _name_ == "_main_":
    root = tk.Tk()
    
    # Set theme (requires tcl/tk 8.5+)
    try:
        root.tk.call('source', 'azure.tcl')
        root.tk.call('set_theme', 'light')
    except:
        pass
    
    app = JITCompilerGUI(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()
