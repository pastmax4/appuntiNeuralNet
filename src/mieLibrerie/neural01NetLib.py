import numpy as np


#----------------------------------------
# Classe Network
# __init__(self , sizes) è il costruttore.
# Come parametri si usa passare sempre self...
# size
# size è il vero parametro in ingresso, una lista.
# Le liste si esprimono con parentesi quadre [].
# Gli indici partono da 0.
# size contiene il numero di neuroni
# nei rispettivi layers.
# Esempio size[2, 3, 1] significa
# layer 0: 2 neuroni (input)
# layer 1: 3 neuroni 
# layer 2: 1 neuroni (ouput)
# 
# weights vedi fig file net01.jpg
# E' una lista di array di numpy.
# Esempio  
#[ 
#   array(
#           [[ 0.27392862, 1.09685052],
#           [ 0.43599092,  0.31820947],
#           [-0.14733133,  0.54429598]]
#      ), 
#   array([[ 1.17393365, -0.65324522, -0.78784323]])
# ]
#
# Prendiamo weights[1]
# weights[1] = array([[ 1.17393365, -0.65324522, -0.78784323]])
# Sono i pesi che collegano il nodo 3 di uscita con i 3 nodi centrali (Hidden layer) 
#
# Passiamo a weights[0], layer 1 e 0.
# layer 0 -> 2 nodi
# layer 1 -> 3 nodi
# matrice 3 x 2
#
# weights[0] = array(
#           [[ 0.27392862, 1.09685052],
#           [ 0.43599092,  0.31820947],
#           [-0.14733133,  0.54429598]]
#      ), 
#
# weights[0][0][0]), weights[0][0][1])  
# [0]    -> il primo indice indica l'array nella lista
# [0][0] -> 0.27392862 
# [0][1] -> 1.09685052
# Sono i pesi del primo neurone del layer 2 collegato ai 2 neuroni di input.
# Nella figura net01.jpg quelli colorati di rosso.
#
# La seconda riga weights[0][1][0]), weights[0][1][1])  -> [ 0.43599092,  0.31820947]
# fornisce i pesi del secondo neurone del leyer 2 con i due 2 neuroni di input.
# Nella figura net01.jpg quelli colorati di verde.
# 
# La terza riga weights[0][2][0]), weights[0][2][1])  -> [-0.14733133,  0.54429598]
# fornisce i pesi del secondo neurone del leyer 2 con i due 2 neuroni di input.
# Nella figura net01.jpg quelli colorati di blu.
# 
# In generale w[n,j,k] 
# n -> leyer
# j -> j neurone del leyer n+1
# k -> k neurone del leyer n  
#---------------------------------------------------------- 

class Network(object):
    def __init__(self , sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

        


#--------------------------------------------------------------------
#### Miscellaneous functions
#--------------------------------------------------------------------
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
