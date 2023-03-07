import numpy as np

A = np.array([2,1,1,1,3,6,4,5,5]).reshape(3,3)
B = np.array([1,0,5,2,1,6,0,3,0]).reshape(3,3)

C = np.dot(A,B)

print("Mnozenie za pomoca np.dot() \n",C)
D=np.matmul(A,B)
E = A@B

print("Mnozenie za pomoca operatora np.matmul() \n",D)
print("Mnozenie za pomoca operatora @ \n",E)

F = np.array([[6,1,1,4],
             [4,-2,5,1],
             [2,8,7,4],
             [2,2,1,3]])
print("Rank of A:", np.linalg.matrix_rank(F))

G = np.array([[6,1,1],
              [4,-2,5],
              [2,8,7]])

print("\nDeterminant of G:",np.linalg.det(G))

H = np.array([[4,3,2,0],
              [-4,-2,-5,1],
              [12,8,-7,4]])

M = np.transpose(H)
print(M)
print("\n")


def multiply_matrices(A, B):
    # sprawdzenie wymiarów macierzy
    if A.shape[1] != B.shape[0]:
        print("Wymiary macierzy nie są zgodne")
        return None

    # macierz wynikowa
    C = np.zeros((A.shape[0], B.shape[1]))

    # mnożenie macierzy za pomocą pętli
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]

    return C

A = np.array([[2, 1, 1], [1, 3, 6], [4, 5, 5]])
B = np.array([[1, 0, 5], [2, 1, 6], [0, 3, 0]])
C = multiply_matrices(A, B)
print(C)

def determinant_3x3(matrix):
    # sprawdzenie wymiarów macierzy
    if len(matrix) != 3 or len(matrix[0]) != 3:
        print("Wymiar macierzy nie jest zgodny.")
        return None

    # obliczenie wyznacznika za pomocą reguły Sarrusa
    det = 0
    for i in range(3):
        det += matrix[0][i] * matrix[1][(i+1)%3] * matrix[2][(i+2)%3]
        det -= matrix[0][i] * matrix[1][(i+2)%3] * matrix[2][(i+1)%3]

    return det

matrix = [[1, 4, 5], [2, 1, 6], [0, 3, 2]]
det = determinant_3x3(matrix)
print("Wyznaczik jest równy: ",det)

def transpose(matrix):
    # inicjalizacja macierzy wynikowej
    result = [[0 for j in range(len(matrix))] for i in range(len(matrix[0]))]

    # transponowanie macierzy
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[j][i] = matrix[i][j]

    return result

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
transposed = transpose(matrix)
print(transposed)