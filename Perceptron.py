import numpy as np

import numba.types as types
from numba import njit, int64, float64
from numba.typed import Dict
from numba.experimental import jitclass

kv_ty_2d = (int64, float64[:, :])

spec = [
    ('nodes_list', int64[:]),
    ('weights', types.DictType(*kv_ty_2d)),
    ('biases' , types.DictType(*kv_ty_2d)),
    ('training_iterations', int64),
    ('guessed', float64[:])
]

@jitclass(spec)
class Perceptron():
    def __init__(self, Nodes, training_iterations):
        # The weights are between -1 and 1 at the beginning
        self.nodes_list = Nodes
        self.weights, self.biases = self.weights_init()
        self.training_iterations = training_iterations
        self.guessed = np.zeros(training_iterations)
        
    def weights_init(self):
        out_w = Dict.empty(*kv_ty_2d)
        out_b = Dict.empty(*kv_ty_2d)

        for i in range(1, len(self.nodes_list)):
            out_w[i] = 2 * np.random.rand(self.nodes_list[i-1], self.nodes_list[i]) - 1 
            out_b[i] = 2 * np.random.rand(1                   , self.nodes_list[i]) - 1 
            
        return out_w, out_b
    
    def train(self, training_input, training_output):

        steps_dict  = Dict.empty(*kv_ty_2d)
        errors_dict = Dict.empty(*kv_ty_2d)
        deltas_dict = Dict.empty(*kv_ty_2d)

        ll = len(self.nodes_list)
        lentr = len(training_input)

        for it in range(self.training_iterations):
            for L0, Llast_Train in zip(training_input, training_output):
            
                steps_dict[0] = L0[:, np.newaxis]

                for i in range(1, ll):
                    steps_dict[i] = step(steps_dict[i-1], self.weights[i], self.biases[i])

                self.guessed[it] += int(np.argmax(steps_dict[ll-1]) == np.argmax(Llast_Train))
                
                errors_dict[ll-1] = Llast_Train[:, np.newaxis] - steps_dict[ll-1]
                deltas_dict[ll-1] = delta(errors_dict[ll-1], steps_dict[ll-1])

                for i in range(len(self.nodes_list)-2,-1,-1):
                    errors_dict[i] = np.dot(self.weights[i+1], deltas_dict[i+1])
                    deltas_dict[i] = delta(errors_dict[i], steps_dict[i])

                for i in range(len(self.nodes_list)-2,-1,-1):
                    self.weights[i+1] += np.dot(steps_dict[i], deltas_dict[i+1].T)
                    tmp = np.ones(len(deltas_dict[i+1]))[None, :]
                    self.biases[i+1] += np.dot(tmp , deltas_dict[i+1]).T
   
    def guess(self, input_data):
        out = input_data
        for i in range(1, len(self.nodes_list)):
            out = step(out, self.weights[i], self.biases[i])
        return out
    
@njit
def delta(error, output_layer):
    return np.multiply(error, norm_deriv(output_layer))

@njit
def step(layer_in, weights, bias):
    return norm( np.dot(layer_in.T, weights) + bias ).T

@njit
def norm(x):
    return 1 / (1 + np.exp(-x))

@njit
def norm_deriv(x):
    return x * (1 - x)