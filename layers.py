import numpy as np

def relu(x):
    return np.maximum(x, 0)


def leaky_relu(x, a=0.3):
    if a <= 0 or a > 1:
        raise ValueError('LeakyReLU: alpha must be between 0 and 1')
    return np.maximum(x, a * x)


def softmax(x):
    '''
    Numerically more stable implementation of
    softmax function. Ref: http://cs231n.github.io/linear-classify/
    '''
    x -= np.max(x)  # maximum value is now 0, so exponentials wont break
    probabilities = np.exp(x) / np.sum(np.exp(x))
    return probabilities


def cross_entropy_loss(y_pred, y_true):
    return -np.log(y_pred[np.argmax(y_true)])   # -negative log probability of the right class


if __name__ == '__main__':
    """
    Test cases
    """
    print('relu and leaky relu:')
    values = np.random.rand(4,5) - .5
    print(values)
    print(relu(values))
    print(leaky_relu(values, .1))

    print('Softmax:')
    x = np.array([1,2,3,4])
    print(x)
    print(softmax(x))

    print('Cross entropy loss:')
    y_pred = np.array([1,2,3,4,5])
    y_true = np.array([0,0,1,0,0])
    print('y_pred: {}, y_true: {}'.format(y_pred, y_true))
    print('loss: {}'.format(cross_entropy_loss(y_pred, y_true)))

