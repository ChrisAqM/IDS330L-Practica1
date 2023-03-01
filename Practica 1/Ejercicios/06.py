
'''
06. Explique cómo combinan estas neuronas las diferentes entradas que reciben.

    Las diferentes entradas se combinan mediante combinación lineal. En particular, 
    la función forward de la clase Neuron realiza esta combinación lineal mediante 
    el producto punto (dot product) entre el vector de entrada (x) y los pesos de 
    la neurona (self.W).
    
    El resultado de esta operación es sumado con un término de sesgo (self.b) para 
    obtener una suma ponderada (z).
'''
