import numpy as np

def xavier_uniform_init(shape):
    bound = np.sqrt(6. / (shape[0] + shape[1]))
    return np.random.uniform(-bound, bound, size=shape).astype('float64')


def xavier_normal_init(shape):
    return np.random.normal(0, np.sqrt(2. / (shape[0] + shape[1])), size=shape).astype('float64')


def he_uniform_init(shape, a=0):
    input_dim = shape[0]
    bound = np.sqrt(6. / ((1. + np.square(a)) * input_dim))
    return np.random.uniform(-bound, bound, size=shape).astype('float64')


def he_normal_init(shape, a=0):
    input_dim = shape[0]
    return np.random.normal(0, np.sqrt(2. / ((1. + np.square(a)) * input_dim)), size=shape).astype('float64')


def random_uniform_init(shape, low=-0.5, high=0.5):
    return np.random.uniform(low, high, size=shape).astype('float64')


def random_normal_init(shape, mean=0, stddev=0.05):
    return np.random.normal(mean, stddev, size=shape).astype('float64')
