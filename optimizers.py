from layers import *

class SGD:
    """docstring for ClassName"""
    def __init__(self, learning_rate = 0.001, momentum = 0.9, nesterov = False):
        self.learning_rate = learning_rate
        self.v = []
        self.momentum = momentum
        self.use_nesterov = nesterov

    def init_shape(shape_list):
        for shape in shape_list:
            self.v.append(np.zeros(shape))

    def optimize(self, params_gradient):
        for i in range(len(params_gradient)):
            parameter, gradient = params_gradient[i]
            if self.use_nesterov:
                dervative = v[i]
                v[i] = self.momentum * v[i] + self.learning_rate * gradient
                dervative = self.momentum * dervative - (1.0 + self.momentum) * v[i]
            else:
                dervative = self.momentum * v[i] - self.learning_rate * gradient
                v[i] = dervative
            parameter += dervative

        


class Adam:
    def __init__(self, learning_rate = 0.001, B1 = 0.9, B2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.B1 = B1
        self.B2 = B2
        self.epsilon = epsilon
        self.mt = []
        self.vt = []
        self.t = 0

    def init_shape(shape_list):
        for shape in shape_list:
            self.mt.append(np.zeros(shape))
            self.vt.append(np.zeros(shape))

    def optimize(self, params_gradient):
        self.t += 1

        for i in range(len(params_gradient)):
            parameter, gradient = params_gradient[i]
            mt[i] *= self.B1
            mt[i] += (1 - self.B1) * gradient
            vt[i] *= self.B2
            vt[i] += (1 - self.B2) * (gradient * gradient)
            mt_hat = mt[i] / (1 - self.B1 ** self.t)
            vt_hat = vt[i] / (1 - self.B2 ** self.t)
            parameter -= self.learning_rate * mt_hat / (np.sqrt(vt_hat) + self.epsilon)




        
        