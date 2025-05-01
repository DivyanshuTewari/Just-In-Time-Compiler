import ply.lex as lex
import re

tokens = (
    'NUMBER', 'IDENTIFIER', 'STRING', 'BOOL',
    'PLUS', 'MINUS', 'MUL', 'DIV',
    'AND', 'OR', 'NOT',
    'EQ', 'NE', 'LT', 'GT', 'LE', 'GE',
    'IF', 'ELSE',
    'LPAREN', 'RPAREN', 'EQUALS', 'SEMICOLON'
)
# reserver tokens defined here
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

    # function called here

lexer = lex.lex()