import pygame
import sys

# Define puzzle grid and clues
puzzle_grid = [
    ['C', 'R', 'O', 'S', 'S', 'W', 'O', 'R', 'D', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
]

clues = {
    'across': {
        1: "Opposite of down",
        4: "Not odd",
        7: "A long narrative poem",
    },
    'down': {
        1: "A brief written or spoken account",
        2: "A place for storing grain",
        3: "Frozen water",
    }
}

# Initialize pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the display
cell_size = 40
window_size = (len(puzzle_grid) * cell_size, len(puzzle_grid[0]) * cell_size)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Crossword Puzzle")

# Function to draw the crossword grid
def draw_grid():
    for i in range(len(puzzle_grid)):
        for j in range(len(puzzle_grid[i])):
            pygame.draw.rect(screen, WHITE, (j * cell_size, i * cell_size, cell_size, cell_size), 1)
            if puzzle_grid[i][j] != ' ':
                font = pygame.font.Font(None, 36)
                text = font.render(puzzle_grid[i][j], True, BLACK)
                screen.blit(text, (j * cell_size + 15, i * cell_size + 10))

# Function to draw clues
def draw_clues():
    font = pygame.font.Font(None, 24)
    for direction, direction_clues in clues.items():
        for number, clue_text in direction_clues.items():
            text = font.render(f"{number}. {clue_text}", True, BLACK)
            if direction == 'across':
                screen.blit(text, (window_size[0] // 2, number * 25))
            else:
                screen.blit(text, (window_size[0] // 2 + 200, number * 25))

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(WHITE)
    draw_grid()
    draw_clues()
    pygame.display.flip()