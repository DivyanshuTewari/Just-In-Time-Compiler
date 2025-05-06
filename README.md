
<p align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/gui-example.png" alt="GUI Screenshot" width="700">
</p>

---

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
