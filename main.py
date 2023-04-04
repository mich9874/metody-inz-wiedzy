import numpy as np
def det(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        det = matrix[0][0] * matrix[1][1] * matrix[2][2]
        det += matrix[0][1] * matrix[1][2] * matrix[2][0]
        det += matrix[0][2] * matrix[1][0] * matrix[2][1]
        det -= matrix[0][2] * matrix[1][1] * matrix[2][0]
        det -= matrix[0][1] * matrix[1][0] * matrix[2][2]
        det -= matrix[0][0] * matrix[1][2] * matrix[2][1]
        return det
    elif len(matrix) == 4:
        det = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * \
              (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) - \
              (matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]) * \
              (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) + \
              (matrix[0][0] * matrix[1][3] - matrix[0][3] * matrix[1][0]) * \
              (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1]) + \
              (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * \
              (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) - \
              (matrix[0][1] * matrix[1][3] - matrix[0][3] * matrix[1][1]) * \
              (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0]) + \
              (matrix[0][2] * matrix[1][3] - matrix[0][3] * matrix[1][2]) * \
              (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0])
        return det
def calculate_rank(matrix):
    """Funkcja obliczająca rząd macierzy"""
    dett = det(matrix)
    if dett != 0:
        return len(matrix)
    else:
        submatrices = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                submatrix = [row[:j] + row[j+1:] for row in matrix[:i] + matrix[i+1:]]
                submatrices.append(submatrix)
        for submatrix in submatrices:
            subdet = det(submatrix)
            if subdet != 0:
                return len(submatrix)
        for row in matrix:
            if any(row):
                return 1
        return 0

matrix = [[1, 1, 5],
          [2, 0, 6],
          [8, 3, 2]]

matrixb = [[3, -1, 1],
           [5, 1, 5],
           [-1, 3, 2]]

matrixc = [[1, 3, -2, 4],
           [1, -1, 3, 5],
           [0, 1, 4, -2],
           [10, -2, 5, 1]]

matrixd = [[2, 8, 3, -4],
           [1, 4, 1, -2],
           [5, 20, 0, -10],
           [-3, -12, -2, 6]]

rank = calculate_rank(matrix)
rankb = calculate_rank(matrixb)
rankc = calculate_rank(matrixc)
rankd = calculate_rank(matrixd)
#A
print("Rząd macierzy A: ",rank)
print("Rząd macierzy numpy A: ", np.linalg.matrix_rank(matrix))
#B
print("Rząd macierzy B: ",rankb)
print("Rząd macierzy numpy B: ", np.linalg.matrix_rank(matrixb))

#C
print("Rząd macierzy C: ",rankc)
print("Rząd macierzy numpy C: ", np.linalg.matrix_rank(matrixc))

#D
print("Rząd macierzy D: ",rankd)
print("Rząd macierzy numpy D: ", np.linalg.matrix_rank(matrixd))