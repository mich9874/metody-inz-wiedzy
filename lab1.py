import numpy as np
import tensorflow as tf

my_vector1 = np.array([1,2,3,4,5])
my_vector2 = np.arange(5)
my_vector3 = np.arange(0,10,2)
my_vector4 = np.linspace(1,np.pi,4)

new_vector = np.append(my_vector2,[1000])
new_vector_insert = np.insert(my_vector2,1,8888)

index = 0
arr2 = np.delete(my_vector1,index)

scalar = 3

vec1 = np.array([1,2,3,4,5])
vec2 = np.array([1,2,3,4,5])
vec3 = np.array([5,4,3,2,1])
vec4 = np.array([0,2,3,0,2])

adding1 = np.add(vec1,vec2)
adding2 = vec1+vec2

subtract1 = np.subtract(vec1,vec3)
subtract2 = vec1-vec3

result = vec1 * scalar

multiply1 = np.multiply(vec1,vec4)
multiply2 = vec1*vec4

divide1 = np.divide(vec1,vec3)
divide2 = vec1/vec3

print("My vector: ", my_vector1)
print("My vector 2: ", my_vector2)
print("My vector 3: ", my_vector3)
print("My vector 4: ", my_vector4)
print("New vector with 1000 at the end: ",new_vector)
print("Vector with value 8888 at the index 1", new_vector_insert)
print("Based array:",my_vector1, "Array with removed value[0]",arr2)
print("After adding using function: ",adding1)
print("After adding without function: ",adding2)
print("After subtracting using function: ",subtract1)
print("After subtracting without function: ",subtract2)
print(result)
print("After multipled using function: ",multiply1)
print("After multipled without function: ",multiply2)
print("After divided using function: ",divide1)
print("After divided without function: ",divide2)

#-------------------------------Macierze------------------------------#

matrix1 = np.array([[1,2,3,4,15], [2,4,6,8,10]])

x = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
X = np.matrix(x)

first_value = x[0][0]
middle_value = x[1][1]
last_value = x[-1][-1]

a = np.diag((1,2,3,4))
zeros = np.zeros((3,3))
ones = np.ones((3,3))

y = zeros.copy()
y[0][2]=6

print(matrix1)
print(x)
print(X)
print("first element in matrix", first_value)
print("middle element in matrix", middle_value)
print("last element in matrix", last_value)
print(a)
print(zeros)
print(ones)
print(y)

#--------------------Tensory-------------------------#

t1 = tf.constant([1,2,3])
t2 = tf.constant([[1.1,2.2,3.3],[4,5,6]])
t3 = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
t4 = tf.constant(["String_one","String_two","String_three"])

# print(t1)
# print("\n")
# print(t2)
# print("\n")
# print(t3)
# print("\n")
# print(t4)
# print("\n")
#--------------Zadania------------------#

vector1 = np.array([1,4,5,6,2,1,5,6,7,0])
index2 = 5
arr3 = np.delete(vector1,index2)
print("Zadanie 1a",arr3)
arr3 = np.append(arr3,[8])
print("Zadanie 1b",arr3)

res = arr3[::-1]
print("Zadanie 1d", res)

