from lark import Lark, Transformer

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
