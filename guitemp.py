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
        tk.Button(btn_frame, text="Run", command=self.run).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Save Code", command=self.save_code).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Load Code", command=self.load_code).pack(side=tk.LEFT, padx=2)

        output_frame = tk.LabelFrame(self.root, text="Output")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.output = tk.Text(output_frame, height=8, font=("Consolas", 12), bg="#f7f7f7", state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True)

        self.statusbar = tk.Label(self.root, text="", anchor=tk.W, relief=tk.SUNKEN)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    def create_tags(self):
        self.editor.tag_configure('keyword', foreground='#0057b7', font=("Consolas", 13, "bold"))
        self.editor.tag_configure('number', foreground='#d35400')
        self.editor.tag_configure('string', foreground='#27ae60')
        self.editor.tag_configure('identifier', foreground='#34495e')

    def on_key_release(self, event=None):
        self.highlight_syntax()

    def highlight_syntax(self):
        code = self.editor.get("1.0", tk.END)
        for tag in ['keyword', 'number', 'string', 'identifier']:
            self.editor.tag_remove(tag, "1.0", tk.END)
        for pattern, tag in SYNTAX_PATTERNS:
            for match in re.finditer(pattern, code):
                start = f"1.0+{match.start()}c"
                end = f"1.0+{match.end()}c"
                self.editor.tag_add(tag, start, end)

    def load_example(self, event=None):
        idx = self.example_combo.current()
        if idx >= 0:
            code = EXAMPLES[idx][1]
            self.editor.delete("1.0", tk.END)
            self.editor.insert("1.0", code)
            self.highlight_syntax()
 
