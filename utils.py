import numpy as np
import config


def shuffle_together(a, b):
    '''
    Shuffles 2 arrays with same length together
    '''
    assert len(a) == len(b) # Make sure they are of same dim
    random_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)


def to_one_hot_labels(labels):
    '''
    Convert labels to one hot representation
    '''
    one_hot_labels = np.zeros((len(labels), config.NUM_CLASSES))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels


def mean_normalisation(data):
    if len(data.shape) != 2:
        raise ValueError('Missing dimension! (n, dim) required')
    mean = np.mean(data, axis=0)
    data_min = np.amin(data, axis=0)
    data_max = np.amax(data, axis=0)
    data = (data - mean) / (data_max - data_min)
    return data


def whitening(data):
    if len(data.shape) != 2:
        raise ValueError('Missing dimension! (n, dim) required')
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    data = (data - mean) / var
    return data


if __name__ == '__main__':
    """
    Test cases
    """
    print('to one hot:')
    labels = np.random.randint(0, 10, (15))
    print(labels)
    print(to_one_hot_labels(labels))

    print('mean norm:')
    # data = np.arange(9).reshape(3, 3)
    data = np.random.randint(0,10, (3,3))
    print(data)
    print(mean_normalisation(data))

    print('data whitening:')
    data = np.random.randint(0,10, (3,3))
    print(data)
    print(whitening(data))

