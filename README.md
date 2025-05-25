<h1 align="center">🔥 JIT Compiler Playground</h1>
<p align="center">
  <b>Dynamic Expressions. Native Performance.</b><br>
  A JIT compiler and interpreter for a mini expression language with full GUI, written in pure Python.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.7+-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/platform-Windows%20x86__64-lightgrey">
  <img src="https://img.shields.io/badge/jit-x86--64%20assembler-informational">
  <img src="https://img.shields.io/badge/gui-built%20with%20Tkinter-orange">
</p>
---

## GUI 

![image](https://github.com/user-attachments/assets/3df670d5-8246-4c67-802e-8badb5783f25)

Grammar

Rule 0     S' -> program
Rule 1     program -> statement
Rule 2     program -> statement program
Rule 3     statement -> assignment
Rule 4     statement -> expression SEMICOLON
Rule 5     statement -> expression
Rule 6     assignment -> IDENTIFIER EQUALS expression SEMICOLON
Rule 7     expression -> IF LPAREN expression RPAREN expression ELSE expression
Rule 8     expression -> expression PLUS expression
Rule 9     expression -> expression MINUS expression
Rule 10    expression -> expression MUL expression
Rule 11    expression -> expression DIV expression
Rule 12    expression -> expression AND expression
Rule 13    expression -> expression OR expression
Rule 14    expression -> expression EQ expression
Rule 15    expression -> expression NE expression
Rule 16    expression -> expression LT expression
Rule 17    expression -> expression GT expression
Rule 18    expression -> expression LE expression
Rule 19    expression -> expression GE expression
Rule 20    expression -> NOT expression
Rule 21    expression -> factor
Rule 22    factor -> BOOL
Rule 23    factor -> NUMBER
Rule 24    factor -> IDENTIFIER
Rule 25    factor -> STRING
Rule 26    factor -> LPAREN expression RPAREN


## 📦 Project Structure

| File | Description |
|------|-------------|
| `gui.py`        | 🎨 Full-featured GUI with syntax highlighting & examples |
| `lexer.py`      | 🔍 Tokenizer using **PLY** |
| `parser.py`     | 🧱 Parser that builds the AST |
| `ast_nodes.py`  | 🌳 AST node classes (Number, String, BinOp, etc.) |
| `evaluator.py`  | 🧠 Pure Python evaluator (interpreter) |
| `codegen.py`    | ⚙️ JIT compiler that emits x86-64 assembly & machine code |

---

## 🛠️ Features

✅ Arithmetic: `+`, `-`, `*`, `/`  
✅ Conditionals: `if (cond) then else`  
✅ Strings and concatenation  
✅ Boolean logic: `true`, `false`, `&&`, `||`, `!`  
✅ Variable assignment and usage  
✅ Native x86-64 execution  
✅ JIT using `ctypes` + `VirtualProtect`  
✅ Built-in code examples  
✅ Static memory buffer for strings  
✅ Works entirely from Python without external dependencies (except `ply`)

---

## 🧪 Try These Examples

| Example | Code |
|--------|------|
| Simple Math | `1 + 2 * 3` |
| Conditional | `if (1 < 2) 42 else 0` |
| Boolean Logic | `true && false` |
| String Concatenation | `"Hello, " + "World!"` |
| Assignment | `x = 5; x * 2` |
| String Variables | `s = "Python"; s + " rocks!"` |

---

## ⚙️ Getting Started

### 🔧 Requirements

- Python 3.7+
- Windows OS (x86-64 only)
- [`ply`](https://pypi.org/project/ply/):  
  ```bash
  pip install ply
