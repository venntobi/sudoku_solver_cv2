from itertools import product
import numpy as np

sudoku_cols = "123456789"
sudoku_rows = "ABCDEFGHI"
numbers = list(range(1,10))

exact_over =  np.zeros((729, 324), dtype="int32")

def create_exact_cover():
    pass



def dancing_links(A, Y, solution):
    if not A:
        yield list(solution)
    else:
        c = min(A, key=lambda c: len(A[c]))
        for r in list(A[c]):
            solution.append(r)
            cols = select(A, Y, r)
            for s in dancing_links(A, Y, solution):
                yield s
            deselect(A, Y, r, cols)
            solution.pop()

def select(A, Y, r):
    cols = []
    for j in Y[r]:
        for i in A[j]:
            for k in Y[i]:
                if k != j:
                    A[k].remove(i)
        cols.append(A.pop(j))
    return cols

def deselect(A, Y, r, cols):
    for j in reversed(Y[r]):
        A[j] = cols.pop()
        for i in A[j]:
            for k in Y[i]:
                if k != j:
                    A[k].add(i)