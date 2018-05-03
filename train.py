import h5py
import numpy as np
import utils
from layers import FullyConnected, ReLU, LeakyReLU, BatchNorm, SoftmaxCrossEntropy, Dropout
from initializers import xavier_uniform_init, random_uniform_init
import config

'''
Model and its output activation and loss function
'''
model = [
    FullyConnected(config.INPUT_DIM, 256, xavier_uniform_init, use_weight_norm=True),
    LeakyReLU(),
    Dropout(0.5),
    FullyConnected(256, 128, xavier_uniform_init, use_weight_norm=True),
    LeakyReLU(),
    Dropout(0.1),
    FullyConnected(128, 64, xavier_uniform_init, use_weight_norm=True),
    LeakyReLU(),
    Dropout(0.1),
    FullyConnected(64, 32, xavier_uniform_init, use_weight_norm=True),
    LeakyReLU(),
    Dropout(0.1),
    FullyConnected(32, 16, xavier_uniform_init, use_weight_norm=True),
    LeakyReLU(),
    Dropout(0.1),
    FullyConnected(16, config.NUM_CLASSES, xavier_uniform_init, use_weight_norm=True)
]

output_gate = SoftmaxCrossEntropy()


def forward_pass(model, x_batch, training=True):
    for layer in model:
        x_batch = layer.forward(x_batch, training=training)
    return x_batch


def backward_pass(model, grad):
    for layer in reversed(model):
        grad = layer.backward(grad)


def update_model(model):
    for layer in model:
        layer.update()  # call layer's own optimizer to make update


def go_through_batches(X, y, num_batches, training):
    total_loss = 0
    batch_acc = 0

    utils.shuffle_together(X, y)

    for idx in range(num_batches):
        start_idx = config.BATCH_SIZE * idx
        end_idx = start_idx + config.BATCH_SIZE
        x_batch = X[start_idx:end_idx, :]
        y_batch = y[start_idx:end_idx, :]

        x_batch = forward_pass(model, x_batch, training=training)

        y_batch_pred = output_gate.softmax(x_batch)
        loss = output_gate.cross_entropy(x_batch, y_batch)
        grad = output_gate.backward()

        total_loss += np.mean(loss)
        batch_acc += utils.get_accuracy(y_batch_pred, y_batch)

        if training:
            backward_pass(model, grad)
            update_model(model)

    # Get average loss and accuracy
    total_loss /= num_batches
    batch_acc /= num_batches
    return total_loss, batch_acc


def read_data(filepath):
    with h5py.File('{}/train_128.h5'.format(filepath),'r') as H:
        X = np.copy(H['data'])

    with h5py.File('{}/train_label.h5'.format(filepath),'r') as H:
        y = np.copy(H['label'])

    return X, y


def main():
    X, y = read_data(config.DATA_PATH)
    y = utils.to_one_hot_labels(y)
    utils.shuffle_together(X, y)

    split = len(X) // 20 * 19
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalise after split
    X_train, X_test = utils.whitening(X_train), utils.whitening(X_test)

    # Try tiny subset
    # X_train = X_train[:config.BATCH_SIZE*4]
    # y_train = y_train[:config.BATCH_SIZE*4]
    # X_test = X_test[:config.BATCH_SIZE*4]
    # y_test = y_test[:config.BATCH_SIZE*4]

    train_size = len(X_train)
    test_size = len(X_test)
    num_batches = train_size // config.BATCH_SIZE
    num_test_batches = test_size // config.BATCH_SIZE

    for e in range(config.NUM_EPOCHS):
        train_loss, train_acc = go_through_batches(X_train, y_train, num_batches, True)
        test_loss, test_acc = go_through_batches(X_test, y_test, num_test_batches, False)
        print("Epoch {:>3} | Loss: {:.6f}, Accuracy: {:.6f} | Test loss: {:.6f}, Test Accuracy: {:.6f}"
                .format(e, train_loss, train_acc, test_loss, test_acc))


if __name__ == '__main__':
    main()

