import numpy as np
import config

def to_one_hot_labels(labels):
    one_hot_labels = np.zeros((len(labels), config.NUM_CLASSES))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels


if __name__ == '__main__':
    """
    Test cases
    """
    labels = np.random.randint(0, 10, (15))
    print(labels)
    print(to_one_hot_labels(labels))