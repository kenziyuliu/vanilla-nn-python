import numpy as np

def to_one_hot_label(labels):
    one_hot_labels = np.zeros((len(labels), labels.max() + 1))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels


if __name__ == '__main__':
    """
    Test cases
    """
    print(to_one_hot_label(np.arange(10)))