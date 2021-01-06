import numpy as np
from time import time

#Test-Sudoku
grid = [[1,0,7,0,0,5,4,0,0],
        [9,0,0,3,7,0,0,0,0],
        [0,2,3,0,8,0,0,0,0],
        [0,9,2,0,0,0,0,7,0],
        [0,7,0,0,6,0,0,1,0],
        [0,6,0,0,0,0,8,9,0],
        [0,0,0,0,4,0,3,6,0],
        [0,0,0,0,3,7,0,0,1],
        [0,0,8,2,0,0,5,0,7]]


def possible(row,col,num,grid):
    """ Check if num can be passed to grid[row][col]"""
    if grid[row][col] != 0:
        return False
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    row0 = (row//3)*3
    col0 = (col//3)*3
    for i in range(3):
        for j in range(3):
            if grid[row0+i][col0+j] == num:
                return False
    return True

def is_solved(grid):
    """ Check if Sudoku is already solved/full"""
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return False
    return True


# def solve(grid):
#     """ Backtracking algorithm to solve Sudoku"""
#     for r in range(9):
#         for c in range(9):
#             if grid[r][c] == 0:
#                 for i in range(1,10):
#                     if possible(r, c, i, grid): #
#                         grid[r][c] = i
#                         if is_solved(grid):
#                             print("Show Solved Sudoku:")
#                             print(grid)
#                             return(grid) 
#                         solve(grid)
#                         grid[r][c] = 0
#                 return 
#     return 
    

def solve(grid):
    """ Backtracking algorithm to solve Sudoku"""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for i in range(1,10):
                    if possible(r, c, i, grid): 
                        grid[r][c] = i 
                        solve(grid)
                    if is_solved(grid):
                        return
                    grid[r][c] = 0
                return



# Returned Vektor mit allen Zahlen, die
# eingetragen werden könnten
# def presolve(row, col):
#     return [i for i in range(1, 10) if possible(row, col, i)]

# def lookup(row):
#     global grid
#     lookup_dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
#     for col in range(9):
#         poss = presolve(row, col)
#         for p in poss:
#             lookup_dict[p] += 1

#     for i in range(1, 10):
#         if lookup_dict[i] == 1:
#             for col in range(9):
#                 if possible(row, col, i):
#                     print(row, col, i)
#                     grid[row][col] = i
#                     return True
#     return False

start = time()
solve(grid)
end = time()
print(end-start)

grid