import numpy as np
import config
from optimizers import get_optimizer
import copy


"""
NOTE:
    - Only use dropout during training
"""

class FullyConnected:
    def __init__(self,
                 input_dim,
                 output_dim,
                 weight_initializer,
                 use_bias=False,
                 use_weight_norm=False,
                 opt=config.OPT,
                 clip_gradients=False):

        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.clip_gradients = clip_gradients    # Option for clipping gradients

        self.optimizer = get_optimizer(opt)
        param_shapes = []                       # For initializing the optimizer

        self.input = None
        self.b = np.zeros(output_dim, dtype='float64')
        self.db = None

        if use_weight_norm:
            # There are `output_dim` number of weight vectors `w` with length `input_dim`
            self.v_shape = (input_dim, output_dim)
            self.g_shape = (output_dim,)
            self.v = weight_initializer(self.v_shape)           # Initialise using the given initialiser
            self.g = np.linalg.norm(self.v, axis=0)
            self.dv = None                                      # No need to initialse gradients
            self.dg = None
            param_shapes.extend([self.v_shape, self.g_shape])   # Shapes for optimizer
        else:
            self.W_shape = (input_dim, output_dim)
            self.W = weight_initializer(self.W_shape)
            self.dW = None
            param_shapes.append(self.W_shape)

        if use_bias:
            param_shapes.append(self.b.shape)
        # Init optimizer using shape of used parameters; e.g. gradient velocities
        self.optimizer.init_shape(param_shapes)


    def get_weight(self):
        ''' Return weights with shape (input_dim x output_dim), depending on weight_norm '''
        if self.use_weight_norm:
            v_norm = np.linalg.norm(self.v, axis=0)
            return self.g * self.v / np.maximum(v_norm, config.EPSILON) # EPSILON for stability
        else:
            return self.W


    def forward(self, input, training=None):
        '''
        Compute forward pass and save input for backprop
        `training` parameter is ignored for conforming with interface
        '''
        self.input = input
        return (self.input @ self.get_weight()) + (self.b if self.use_bias else 0)


    def backward(self, backproped_grad):
        '''
        Use back-propagated gradient (n x out_dim) to compute this layer's gradient
        This function saves dW and returns d(Loss)/d(input)
        '''
        assert backproped_grad.shape == (self.input.shape[0], self.get_weight().shape[1])
        dweights = self.input.T @ backproped_grad       # shape = (input_dim, output_dim)

        if self.use_weight_norm:
            v_norm = np.maximum(np.linalg.norm(self.v, axis=0), config.EPSILON)  # Clip for numerical stability
            self.dg = np.sum(dweights * self.v / v_norm, axis=0)      # Use sum since g was broadcasted
            self.dv = (self.g / v_norm * dweights) - (self.g * self.dg / np.square(v_norm) * self.v)
        else:
            self.dW = dweights

        if self.use_bias:
            self.db = np.sum(backproped_grad, axis=0)   # Sum gradient since bias was broadcasted

        dinput = backproped_grad @ self.get_weight().T  # shape = (batch, input_dim)
        return dinput


    def update(self):
        ''' Update the weights using the optimizer using the latest weights/gradients '''
        params_gradient = []

        if self.use_weight_norm:
            if self.clip_gradients:
                self.dv = self.clip_grad(self.dv)
                self.dg = self.clip_grad(self.dg)
            params_gradient.extend([(self.v, self.dv), (self.g, self.dg)])
        else:
            if self.clip_gradients:
                self.dW = self.clip_grad(self.dW)
            params_gradient.append((self.W, self.dW))

        if self.use_bias:
            if self.clip_gradients:
                self.db = self.clip_grad(self.db)
            params_gradient.append((self.b, self.db))

        # Let the optimizer to do optimization
        self.optimizer.optimize(params_gradient)


    def clip_grad(self, gradient, mingrad=-1., maxgrad=1.):
        ''' Clip gradients in a range to prevent explosion '''
        return np.maximum(mingrad, np.minimum(maxgrad, gradient))



class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input, training=None):
        ''' input.shape = output.shape = (batch x input_dims) '''
        self.input = input
        return np.maximum(input, 0)

    def backward(self, backproped_grad):
        deriv = np.where(self.input < 0, 0, 1)
        return backproped_grad * deriv

    def update(self):
        pass    # Nothing to update



class LeakyReLU:
    def __init__(self, alpha=0.1):
        if alpha <= 0 or alpha > 1:
            raise ValueError('LeakyReLU: alpha must be between 0 and 1')
        self.alpha = alpha
        self.input = None

    def forward(self, input, training=None):
        ''' input.shape = output.shape = (batch x input_dims) '''
        self.input = input
        return np.maximum(input, self.alpha * input)

    def backward(self, backproped_grad):
        ''' Compute gradient of LeakyReLU and backprop '''
        deriv = np.where(self.input < 0, self.alpha, 1)
        return backproped_grad * deriv

    def update(self):
        pass    # Nothing to update



class Dropout:
    def __init__(self, drop_rate):
        if drop_rate < 0 or drop_rate >= 1:
            raise ValueError('Dropout: dropout rate must be >= 0 and < 1')
        self.retain_rate = 1. - drop_rate
        self.mask = None
        self.input = None

    def forward(self, input, training):
        ''' Drop units according to the drop_rate; rescale weights as needed '''
        if not training:
            return input    # During test time, no dropout required

        self.input = input
        self.mask = np.random.binomial(1, self.retain_rate, input.shape)
        self.mask = self.mask / self.retain_rate   # divide rate, so no change for prediction
        return input * self.mask

    def backward(self, backproped_grad):
        ''' Mask gradients according to drop mask; rescale gradients as needed '''
        return backproped_grad * self.mask / self.retain_rate # divide rate so no change for prediction

    def update(self):
        pass    # Nothing to update



class BatchNorm:
    def __init__(self, input_dim, avg_decay=0.9, epsilon=1e-3, opt=config.OPT):
        self.gamma = np.ones(input_dim, dtype='float64')
        self.beta = np.zeros(input_dim, dtype='float64')
        self.d_gamma = None
        self.d_beta = None
        self.running_avg_mean = np.zeros(input_dim, dtype='float64')
        self.running_avg_std = np.zeros(input_dim, dtype='float64')
        self.avg_decay =  avg_decay
        self.epsilon = epsilon
        self.input_hat = None
        self.std = None
        self.optimizer = get_optimizer(opt)

        shape_list = [self.gamma.shape, self.beta.shape]
        self.optimizer.init_shape(shape_list)

    def forward(self, input, training):
        if training:
            self.std = np.sqrt(np.var(input, axis=0) + self.epsilon)
            mean = np.mean(input, axis=0)
            self.input_hat = (input - mean) / self.std
            self.running_avg_mean = self.avg_decay * self.running_avg_mean + (1 - self.avg_decay) * mean
            self.running_avg_std = self.avg_decay * self.running_avg_std + (1 - self.avg_decay) * self.std
            return self.gamma * self.input_hat + self.beta
        else:
            input_hat = (input - self.running_avg_mean) / self.running_avg_std
            return self.gamma * input_hat + self.beta

    def backward(self, backproped_grad):
        d_xhat = backproped_grad * self.gamma
        numerator = len(self.input_hat) * d_xhat - np.sum(d_xhat, axis=0)
        numerator -= self.input_hat * np.sum(d_xhat * self.input_hat, axis=0)
        dx = (1. / len(self.input_hat)) * numerator / self.std
        self.d_gamma = np.sum(backproped_grad * self.input_hat, axis=0)
        self.d_beta = np.sum(backproped_grad, axis=0)
        return dx

    def update(self):
        params_gradient = [(self.gamma, self.d_gamma), (self.beta, self.d_beta)]
        self.optimizer.optimize(params_gradient)



class SoftmaxCrossEntropy:
    def __init__(self):
        self.input = None
        self.y_pred = None
        self.y_true = None

    def softmax(self, input, training=None):
        ''' Compute the softmax (prediction) given input '''
        input -= np.max(input, axis=-1, keepdims=True)  # For numerical stability
        exps = np.exp(input)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def cross_entropy(self, y_pred, y_true):
        '''
        Compute CrossEntropy loss given predictions and labels;
        Calls self.softmax() for prediction
        '''
        y_pred = np.copy(y_pred)        # Copy to ensure not corrupting original predictions
        # negative log likelihood of the right class
        logs = -np.log(y_pred[range(len(y_pred)), np.argmax(y_true, axis=-1)])
        loss = np.mean(logs)            # Real valued average loss over batch
        self.y_true = y_true
        self.y_pred = y_pred
        return loss

    def backward(self):
        '''
        Compute gradient of loss directly with respect to self.input (batch before softmax)
        across Softmax and CrossEntropy; this way is more numerically stable
        '''
        grad = self.y_pred
        # gradient = y_pred - y_true, and y_true == 1 only for the right classes
        grad[range(len(grad)), np.argmax(self.y_true, axis=-1)] -= 1
        grad /= len(grad)
        return grad

    def update(self):
        pass    # Nothing to update


if __name__ == '__main__':
    """
    Test cases
    """
    # print('relu and leaky relu:')

    input_val = np.random.randn(1,2)
    print('input_val:\n',input_val)

    grad_val = np.random.randn(1,3)
    print('grad_val:\n',grad_val)

    import initializers
    FC = FullyConnected(2,3, initializers.xavier_normal_init, use_weight_norm=True)
    FC_weight = FC.get_weight()
    # FC_weight = FC.W
    print('FC_weight:\n', FC_weight)
    print('FC_forward:\n', FC.forward(input_val))
    print('FC backward:\n',FC.backward(grad_val))
    print('FC.dv\n',FC.dv)
    print('FC.dg\n',FC.dg)

    # print('FC_length\n',np.linalg.norm(FC_weight.W, axis=0))
    print('relu:\n', ReLU().forward(input_val))
    print('leakyrelu:\n', LeakyReLU(.1).forward(input_val))
    print()
    print('Softmax:')
    sce = SoftmaxCrossEntropy()
    x = np.expand_dims(np.array([1,1,1,2], dtype='float64'), axis=0)  # Mock the batch dimension
    print(x)
    print(sce.softmax(x))
    print()
    print('Cross entropy loss:')
    y_pred = np.expand_dims(np.array([.1, .1, .1, .1, .1]), axis=0)
    y_true = np.expand_dims(np.array([0,0,1,0,0]), axis=0)
    print('y_pred:\n {} \ny_true:\n {}'.format(y_pred, y_true))
    print('loss:\n {}'.format(sce.cross_entropy(y_pred, y_true)))
    print()
    print('Dropout:')
    x = np.arange(30).reshape(5, 6)
    print(x)
    print(Dropout(0.5).forward(x, training=True))



