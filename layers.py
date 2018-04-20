import numpy as np

def relu(x):
    return np.maximum(x, 0)

def leaky_relu(x, a=0.3):
    if a <= 0 or a > 1:
        raise ValueError('LeakyReLU: alpha must be between 0 and 1')
    return np.maximum(x, a * x)


if __name__ == '__main__':
    """
    Test cases
    """
    values = np.random.rand(4,5) - .5
    print(values)
    print(relu(values))
    print(leaky_relu(values, .1))


