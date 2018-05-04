import numpy as np
import h5py
import utils
import config
from layers import FullyConnected, ReLU, LeakyReLU, BatchNorm, SoftmaxCrossEntropy, Dropout
from initializers import xavier_uniform_init, random_uniform_init
from models import good_model
from models import softmax_crossentropy as sce_gate

'''
Helper methods
'''
def load_training_data(filepath):
    ''' Read the training data from the given file path '''
    with h5py.File('{}/train_128.h5'.format(filepath),'r') as H:
        X = np.copy(H['data'])

    with h5py.File('{}/train_label.h5'.format(filepath),'r') as H:
        y = np.copy(H['label'])

    return X, y


def load_test_data(filepath):
    ''' Read the testing data without labels from given path '''
    with h5py.File('{}/test_128.h5'.format(filepath), 'r') as H:
        X = np.copy(H['data'])

    return X


def forward_pass(model, x_batch, training):
    ''' Given model, forward pass the batch '''
    for layer in model:
        x_batch = layer.forward(x_batch, training=training)
    return x_batch


def backward_pass(model, grad):
    ''' Backward pass the gradients through model while layers save their own gradients '''
    for layer in reversed(model):
        grad = layer.backward(grad)


def update_model(model):
    ''' Update each layer in model by calling their own optimizers '''
    for layer in model:
        layer.update()


def go_through_by_batch(model, X, y, training):
    assert len(X) == len(y)     # Sanity check
    total_loss = total_acc = 0
    num_batches = len(X) // config.BATCH_SIZE

    for idx in range(num_batches):
        # Get mini-batches
        start_idx = config.BATCH_SIZE * idx
        end_idx = start_idx + config.BATCH_SIZE
        x_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Forward pass through the model
        x_batch = forward_pass(model, x_batch, training=training)

        y_batch_pred = sce_gate.softmax(x_batch)
        loss = sce_gate.cross_entropy(y_batch_pred, y_batch)
        grad = sce_gate.backward()

        total_loss += np.mean(loss)
        total_acc += utils.get_accuracy(y_batch_pred, y_batch)

        if training:
            backward_pass(model, grad)
            update_model(model)

    # Get average loss and accuracy
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def model_train(model, X_train, y_train, X_val, y_val):
    print('\nTraining on {} examples, validating on {} examples\n'.format(len(X_train), len(X_val)))

    # Training and validate on the number of epochs
    for e in range(config.NUM_EPOCHS):
        train_loss, train_acc = go_through_by_batch(model, X_train, y_train, training=True)
        val_loss, val_acc = go_through_by_batch(model, X_val, y_val, training=False)

        print("Epoch {:>3} | ".format(e) +
              "Train Loss: {:.5f}, Train Accuracy: {:.5f} | ".format(train_loss, train_acc) +
              "Val Loss: {:.5f}, Val Accuracy: {:.5f}".format(val_loss, val_acc))


def model_evaluate(model, X_test, y_test):
    print('\nEvaluating model on {} unseen examples from training set...'.format(len(X_test)))
    # Evaluate model using part of unseen data from the training set
    test_loss, test_acc = go_through_by_batch(model, X_test, y_test, training=False)

    print('\nTest Loss: {:.5f} | Test Accuracy: {:.5f}\n'.format(test_loss, test_acc))


def model_predict(model, X):
    print('\nMaking predictions on {} testing data...'.format(len(X)))

    y_pred = np.zeros(len(X))   # Space for storing predictions
    num_batches = len(X) // config.BATCH_SIZE

    for i in range(num_batches):
        start_pos = i * config.BATCH_SIZE
        end_pos = start_pos + config.BATCH_SIZE if i < (num_batches - 1) else len(X)
        # Get batch and go through model in batch
        X_batch = X[start_pos:end_pos]
        X_batch = forward_pass(model, X_batch, training=False)
        y_batch_pred = sce_gate.softmax(X_batch)
        y_batch_pred = utils.onehot_vector_to_label(y_batch_pred)   # Convert to labels before storng
        y_pred[start_pos:end_pos] = y_batch_pred                    # Store data

    return y_pred


def main():
    X, y = load_training_data(config.DATA_PATH)
    y = utils.label_to_onehot_vector(y, config.NUM_CLASSES)

    # Shuffle data each time before training
    utils.shuffle_together(X, y)

    data_split = { 'train': 57000, 'val': 1500, 'test': 1500 } # Data split of train/val/test
    split = (data_split['train'],
             data_split['train'] + data_split['val'],
             data_split['train'] + data_split['val'] + data_split['test'])

    y_train, y_val, y_test = y[:split[0]], y[split[0]:split[1]], y[split[1]:split[2]]
    X_train, X_val, X_test = X[:split[0]], X[split[0]:split[1]], X[split[1]:split[2]]

    # Preprocess: normalise after split
    X_train = utils.standardise(X_train)
    X_val = utils.standardise(X_val)
    X_test = utils.standardise(X_test)

    # Train and Test
    model_train(good_model, X_train, y_train, X_val, y_val)
    model_evaluate(good_model, X_test, y_test)

    # Load testing data for marking accuracy and make predictions
    X_test = load_test_data(config.DATA_PATH)
    y_test_pred = model_predict(good_model, X_test)

    # Save prediction to file
    with h5py.File('{}/Predicted_labels.h5'.format(config.DATA_PATH),'w') as H:
        H.create_dataset('label', data=y_test_pred)

    print('Predictions for {} testing data saved to {}/Predicted_labels.h5 with dataset name "label"\n'
            .format(len(X_test), config.DATA_PATH))


if __name__ == '__main__':
    main()

