import tkinter as tk
from gui import JITCompilerGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = JITCompilerGUI(root)
    root.mainloop()