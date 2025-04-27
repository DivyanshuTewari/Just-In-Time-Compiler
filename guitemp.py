import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
from parser import *
from evaluator import eval_program, is_string_node
from codegen import allocate_static_string , compile_ast

SYNTAX_PATTERNS = [
    (r'\b(if|else|true|false)\b', 'keyword'),
    (r'\b\d+\b', 'number'),
    (r'"[^"]*"', 'string'),
    (r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'identifier'),
]

EXAMPLES = [
    ("Simple Math", "1 + 2 * 3"),
    ("If/Else", "if (1 < 2) 42 else 0"),
    ("Boolean", "true && false"),
    ("String", '"Hello, " + "World!"'),
    ("Assignment", "x = 5; x * 2"),
    ("String Variable", 's = "Python"; s + " rocks!"'),
    ("Complex Math", "8*3*6*4*7-1-2-4+4+6*24*213*54"),
    ("Boolean Logic", "a = true; b = false; a && b"),
]

class JITCompilerGUI:
    def __init__(self, root):
        self.root = root
        root.title("JIT Compiler Playground")
        root.geometry("900x600")

        self.create_widgets()
        self.create_tags()
        self.status("Ready.")

    def create_widgets(self):
        editor_frame = tk.Frame(self.root)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.editor = tk.Text(editor_frame, wrap=tk.WORD, font=("Consolas", 13), height=15, undo=True)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.editor.bind("<KeyRelease>", self.on_key_release)

        scrollbar = tk.Scrollbar(editor_frame, command=self.editor.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.editor.config(yscrollcommand=scrollbar.set)

        ex_frame = tk.Frame(self.root)
        ex_frame.pack(fill=tk.X, padx=8)
        tk.Label(ex_frame, text="Examples:").pack(side=tk.LEFT)
        self.example_combo = ttk.Combobox(ex_frame, values=[desc for desc, code in EXAMPLES], state="readonly")
        self.example_combo.pack(side=tk.LEFT, padx=5)
        self.example_combo.bind("<<ComboboxSelected>>", self.load_example)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        tk
