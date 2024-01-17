import tkinter as tk

class CrosswordGUI:
    def __init__(self, clues, grid):
        self.clues = clues
        self.grid = grid
        self.current_cell = None

        self.root = tk.Tk()
        self.root.title("Crossword Puzzle")

        self.create_grid()

    def create_grid(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                cell_value = self.grid[i][j]
                if cell_value == -1:
                    label = tk.Label(self.root, text="", width=2, height=1, bg="black")
                elif cell_value == 0:
                    label = tk.Label(self.root, text="", width=2, height=1, relief="solid")
                    label.bind("<Button-1>", lambda event, i=i, j=j: self.on_click(i, j))
                else:
                    label = tk.Label(self.root, text=str(cell_value), width=2, height=1, relief="solid")

                label.grid(row=i, column=j)

        self.root.mainloop()

    def on_click(self, i, j):
        if self.current_cell:
            self.current_cell.destroy()

        self.current_cell = tk.Entry(self.root, width=2)
        self.current_cell.grid(row=i, column=j)
        self.current_cell.focus_set()

# Example usage:
clues = {'across': ['1. First clue', '2. Second clue'], 'down': ['3. Third clue', '4. Fourth clue']}
grid = [
    [1, -1, 2, -1],
    [0, 0, 0, -1],
    [3, 0, 0, 4],
    [0, -1, -1, -1]
]

crossword_gui = CrosswordGUI(clues, grid)