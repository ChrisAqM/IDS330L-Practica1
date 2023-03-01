from neuron import Neuron, step_function
import numpy as np

# 02. Generar cinco nuevas entradas multiplicando por 2 el valor de las entradas y probarlas con el perceptrón.

# Perceptron input size:
input_size = 5

# Instantiating the perceptron:
perceptron = Neuron(num_inputs=input_size,activation_function=step_function)

print("Perceptron's random weights = {}, and random bias = {}".
      format(perceptron.W, perceptron.b))

x = np.random.rand(input_size).reshape(1, input_size)
x = np.dot(x, 2)  # Producto punto del arreglo aleatorio 
print("Input vector : {}".format(x))

y = perceptron.forward(x)
print("Perceptron's output value given `x` : {}".format(y))