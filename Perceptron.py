import numpy as np

class Perceptron():
    def __init__(self, Nodes):
        # The weights are between -1 and 1 at the beginning
        self.nodes_list = Nodes
        self.weights = self.weights_init()
        self.guessed = {}
        
    def weights_init(self):
        out = {}
        for i in range(1, len(self.nodes_list)):
            w = np.array(2 * np.random.rand(self.nodes_list[i], self.nodes_list[i-1]) - 1)
            b = np.array(2 * np.random.rand(self.nodes_list[i], 1                   ) - 1)
            key_1 = 'w{}'.format(i)
            key_2 = 'b{}'.format(i)
            out[key_1] = w
            out[key_2] = b
        return out
    
    def train(self, training_input, training_output, training_iterations):
        for it in range(training_iterations):
            key_t = 't{}'.format(it)
            self.guessed[key_t] = 0
            for L0, Llast_Train in zip(training_input, training_output):
                L0 = np.array(L0, ndmin=2)
                
                steps_dict  = {}            
                prev_key_L = 'L0'
                steps_dict[prev_key_L] = L0
                for i in range(1,len(self.nodes_list)):
                    key_L = 'L{}'.format(i)
                    key_w = 'w{}'.format(i)
                    key_b = 'b{}'.format(i)
                    steps_dict[key_L] = self.step(steps_dict[prev_key_L], self.weights[key_w], self.weights[key_b])
                    
                    prev_key_L = key_L
                
                self.guessed[key_t] += int(np.argmax(steps_dict[key_L]) == np.argmax(Llast_Train))
                
                errors_dict = {}
                deltas_dict = {}
                
                prev_key_e = 'e{}'.format(len(self.nodes_list)-1)    
                prev_key_d = 'd{}'.format(len(self.nodes_list)-1)    
                errors_dict[prev_key_e] = Llast_Train - steps_dict[key_L]
                deltas_dict[prev_key_d] = self.delta(errors_dict[prev_key_e], steps_dict[key_L])
                for i in range(len(self.nodes_list)-2,-1,-1):
                    key_e = 'e{}'.format(i)
                    key_d = 'd{}'.format(i)
                    key_w = 'w{}'.format(i+1)
                    key_L = 'L{}'.format(i)
                    errors_dict[key_e] = np.dot(deltas_dict[prev_key_d], self.weights[key_w])
                    deltas_dict[key_d] = self.delta(errors_dict[key_e], steps_dict[key_L])
                    
                    prev_key_d = key_d
                
                for i in range(len(self.nodes_list)-2,-1,-1):
                    key_w = 'w{}'.format(i+1)
                    key_L = 'L{}'.format(i)
                    key_d = 'd{}'.format(i+1)
                    key_b = 'b{}'.format(i+1)
                    self.weights[key_w] += np.dot(steps_dict[key_L].T, deltas_dict[key_d]).T
                    self.weights[key_b] += np.dot(1, deltas_dict[key_d]).T
            self.guessed[key_t] = self.guessed[key_t] / len(training_input)
            print(f'In the epoch number {it+1}, {round(self.guessed[key_t]*100, 2)} percent of the training inputs were guessed right.')

        
        
    def step(self, layer_in, weights, bias):
        layer_out = self.norm( np.dot(layer_in , weights.T)  + bias.T )
        return layer_out
    
    def guess(self, input_data):
        out = input_data
        for i in range(1,len(self.nodes_list)):
            key_w = 'w{}'.format(i)
            key_b = 'b{}'.format(i)
            out = self.step(out, self.weights[key_w], self.weights[key_b])
        return out
    
    def delta(self, error, output_layer):
        return np.multiply(error, self.norm_deriv(output_layer))

    def norm(self, x):
        return 1 / (1 + np.exp(-x))

    def norm_deriv(self, x):
        return x * (1 - x)