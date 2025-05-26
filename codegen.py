import ctypes
import struct
import re
from parser import *
from ast_nodes import *
from evaluator import *

STATIC_STRING_BUFFER_SIZE = 4096
_static_string_buffer = ctypes.create_string_buffer(STATIC_STRING_BUFFER_SIZE)
_static_string_next = 0

_label_counter = [0]

def new_label():
    _label_counter[0] += 1
    return f"L{_label_counter[0]}"

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
            code.append("setg al")
            code.append("movzx rax, al")
        elif node.op in ('>', 'GT'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setl al")
            code.append("movzx rax, al")
        elif node.op in ('<=', 'LE'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setge al")
            code.append("movzx rax, al")
        elif node.op in ('>=', 'GE'):
            code += compile_ast(node.left, context, 'rax')
            code.append("push rax")
            code += compile_ast(node.right, context, 'rax')
            code.append("pop rcx")
            code.append("cmp rcx, rax")
            code.append("setle al")
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
    from lexer import lexer
    from parser import parser
    
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
        elif isinstance(node, list):
            for n in node:
                collect_strings(n)
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

 
