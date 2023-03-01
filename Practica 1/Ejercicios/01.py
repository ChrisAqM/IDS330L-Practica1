from neuron import Neuron, step_function
import numpy as np

# 01. Generar cuatro nuevas entradas aleatorias y probarlas con el perceptrón

# Perceptron input size:
input_size = 4

# Instantiating the perceptron:
perceptron = Neuron(num_inputs=input_size, activation_function=step_function)

print("Perceptron's random weights = {}, and random bias = {}".
      format(perceptron.W, perceptron.b))

x = np.random.rand(input_size).reshape(1, input_size)
print("Input vector : {}".format(x))

y = perceptron.forward(x)
print("Perceptron's output value given `x` : {}".format(y))