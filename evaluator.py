from ast_nodes import *

def is_string_node(node, context):
    """Check if a node evaluates to a string"""
    try:
        eval_string_node(node, context)
        return True
    except:
        return False



def _handle_assignment(node, context):
    """Handle variable assignment"""
    varname = node[1]
    value_node = node[2]
    
    # First try numeric evaluation
    try:
        value = eval_numeric_node(value_node, context)
        context['variables'][varname] = ('number', value)
        context['string_vars'].discard(varname)
        return
    except:
        pass
    
    # Then try string evaluation
    try:
        value = eval_string_node(value_node, context)
        context['variables'][varname] = ('string', value)
        context['string_vars'].add(varname)
        return
    except Exception as e:
        raise RuntimeError(f"Assignment failed for {varname}: {str(e)}")

def _eval_node(node, context):
    """Evaluate a single node with proper type fallback"""
    try:
        return eval_numeric_node(node, context)
    except:
        try:
            return eval_string_node(node, context)
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

def eval_numeric_node(node, context):
    """Evaluate a numeric expression"""
    if isinstance(node, Number):
        return node.value
    elif isinstance(node, Bool):
        return int(node.value)
    elif isinstance(node, Identifier):
        if node.name not in context['variables']:
            raise RuntimeError(f"Undefined variable '{node.name}'")
        var_type, var_value = context['variables'][node.name]
        if var_type != 'number':
            raise RuntimeError(f"Variable '{node.name}' is not a number")
        return var_value
    elif isinstance(node, BinOp):
        left = eval_numeric_node(node.left, context)
        right = eval_numeric_node(node.right, context)
        if node.op == '+': return left + right
        if node.op == '-': return left - right
        if node.op == '*': return left * right
        if node.op == '/': return left // right
        if node.op == '&&': return int(bool(left) and bool(right))
        if node.op == '||': return int(bool(left) or bool(right))
        if node.op == '==': return int(left == right)
        if node.op == '!=': return int(left != right)
        if node.op == '<': return int(left < right)
        if node.op == '>': return int(left > right)
        if node.op == '<=': return int(left <= right)
        if node.op == '>=': return int(left >= right)
    elif isinstance(node, UnOp):
        if node.op == '-':
            return -eval_numeric_node(node.val, context)
        if node.op == '!':
            return int(not eval_numeric_node(node.val, context))
    elif isinstance(node, IfElse):
        cond = eval_numeric_node(node.cond, context)
        return eval_numeric_node(node.then_expr if cond else node.else_expr, context)
    raise RuntimeError("Not a numeric node")

def eval_string_node(node, context):
    """Evaluate a string expression"""
    if isinstance(node, String):
        return node.value
    elif isinstance(node, Identifier):
        if node.name not in context['variables']:
            raise RuntimeError(f"Undefined variable '{node.name}'")
        var_type, var_value = context['variables'][node.name]
        if var_type != 'string':
            raise RuntimeError(f"Variable '{node.name}' is not a string")
        return var_value
    elif isinstance(node, BinOp) and node.op == '+':
        left = eval_string_node(node.left, context)
        right = eval_string_node(node.right, context)
        if left.endswith(b'\x00'):
            left = left[:-1]
        return left + right
    elif isinstance(node, IfElse):
        cond = eval_numeric_node(node.cond, context)
        return eval_string_node(node.then_expr if cond else node.else_expr, context)
    raise RuntimeError("Not a string node")