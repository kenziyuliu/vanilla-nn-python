import h5py
import numpy as np
import utils
from layers import FullyConnected, ReLU, LeakyReLU, Softmax, CrossEntropyLoss
import initializers
import config
import time

# model = [
#     layers.FullyConnected(config.INPUT_DIM, 256, weight_initializer=initializers.xavier_uniform_init),
#     layers.LeakyReLU(alpha=0.1),
#     layers.FullyConnected(256, 128, weight_initializer=initializers.xavier_uniform_init),
#     layers.LeakyReLU(),
#     layers.FullyConnected(128, 64, weight_initializer=initializers.xavier_uniform_init),
#     layers.ReLU(),
#     layers.FullyConnected(64, config.NUM_CLASSES, weight_initializer=initializers.xavier_uniform_init),
#     layers.Softmax()
# ]

model = [
    FullyConnected(config.INPUT_DIM, 64, initializers.xavier_uniform_init, use_bias=True),
    # LeakyReLU(),
    FullyConnected(64, config.NUM_CLASSES, initializers.xavier_uniform_init),
    Softmax()
]


def forward_pass(model, x_batch, training=True):
    for layer in model:
        x_batch = layer.forward(x_batch, training=training)
    return x_batch

def backward_pass(model, grad):
    for layer in reversed(model):
        grad = layer.backward(grad)

def go_through_batches(num_batches, training):
    total_loss = batch_acc = 0

    x_data = X_train if training else X_test
    y_data = y_train if training else y_test

    utils.shuffle_together(x_data, y_data)

    for idx in range(num_batches):
        start_idx = config.BATCH_SIZE * idx
        end_idx = start_idx + config.BATCH_SIZE
        x_batch = x_data[start_idx:end_idx, :]
        y_batch = y_data[start_idx:end_idx, :]

        y_batch_pred = forward_pass(model, x_batch, training=training)

        # print('xbatch: {}, ybatch: {}, ybatchpred: {}'.format(x_batch.shape, y_batch.shape, y_batch_pred.shape))

        loss_obj = CrossEntropyLoss()
        loss = loss_obj.forward(y_batch_pred, y_batch)
        grad = loss_obj.backward()

        if training:
            backward_pass(model, grad)
            for layer in model:
                layer.update() # call layer's own optimizer to make update

        total_loss += np.mean(loss)
        batch_acc += utils.get_accuracy(y_batch_pred, y_batch)

        # time.sleep(10)

    total_loss /= num_batches
    batch_acc /= num_batches
    return total_loss, batch_acc


with h5py.File('../train_128.h5','r') as H:
    X = np.copy(H['data'])

with h5py.File('../train_label.h5','r') as H:
    y = np.copy(H['label'])

X = utils.whitening(X)
y = utils.to_one_hot_labels(y)

utils.shuffle_together(X, y)

split = len(X) // 20 * 19
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_size = len(X_train)
test_size = len(X_test)
num_batches = train_size // config.BATCH_SIZE
num_test_batches = test_size // config.BATCH_SIZE

for e in range(config.NUM_EPOCHS):
    train_loss, train_acc = go_through_batches(num_batches, True)
    test_loss, test_acc = go_through_batches(num_test_batches, False)
    print("Epoch {} | Loss: {:.6f}, Accuracy: {:.6f} | Test loss: {:.6f}, Test Accuracy: {:.6f}"
            .format(e, train_loss, train_acc, test_loss, test_acc))
