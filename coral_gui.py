import tkinter as tk


class MainApplication(tk.Frame):
    def __init__(self):
        pass



if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
