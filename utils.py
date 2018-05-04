import numpy as np
import config


def shuffle_together(a, b):
    ''' Shuffles 2 arrays with same length together '''
    assert len(a) == len(b)                 # Sanity check
    random_state = np.random.get_state()    # Store random state s.t. 2 shuffles are the same
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)


def label_to_onehot_vector(labels, num_classes):
    ''' Convert labels to one hot representation '''
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels


def onehot_vector_to_label(vectors):
    ''' Convert onehot vectors to labels '''
    return np.argmax(vectors, axis=-1)


def mean_normalise(data):
    ''' Normalise data to 0 mean and unit range '''
    if len(data.shape) != 2:
        raise ValueError('Missing dimension! (n, dim) required')
    mean = np.mean(data, axis=0)
    data_min = np.amin(data, axis=0)
    data_max = np.amax(data, axis=0)
    data = (data - mean) / (data_max - data_min)
    return data


def standardise(data):
    if len(data.shape) != 2:
        raise ValueError('Missing dimension! (n, dim) required')
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    data = (data - mean) / stddev
    return data


def get_accuracy(y_pred, y_true):
    return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))


if __name__ == '__main__':
    """
    Test cases
    """
    print('to one hot:')
    labels = np.random.randint(0, 10, (15))
    print(labels)
    one_hot = label_to_onehot_vector(labels, config.NUM_CLASSES)
    print(one_hot)

    print('onehot to label')
    predictions = np.random.rand(10, 10)
    reverted_labels = onehot_vector_to_label(predictions)
    print(predictions)
    print(reverted_labels)
    # print(np.array_equal(labels, reverted_labels))

    print('mean norm:')
    # data = np.arange(9).reshape(3, 3)
    data = np.random.randint(0,10, (3,3))
    print(data)
    print(mean_normalise(data))

    print('data whitening:')
    data = np.random.randint(0,10, (3,3))
    print(data)
    print(standardise(data))
