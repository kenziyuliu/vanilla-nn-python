import numpy as np
import h5py
import time
import datetime
import utils
import config
from layers import FullyConnected, ReLU, LeakyReLU, BatchNorm, SoftmaxCrossEntropy, Dropout
from initializers import xavier_uniform_init, random_uniform_init
from models import softmax_crossentropy as sce_gate
from models import he_and_relu, xavier_and_lrelu


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
    ''' Given training examples, pass them forward/backward through the network in batches '''
    # If no examples, return invalid loss and accuracy
    if len(X) == 0:
        return float('inf'), float('inf')

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
    ''' Train and validate the model on the given examples '''
    print('\nAt {}, Training on {} examples, validating on {} examples\n'
        .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(X_train), len(X_val)))

    # Training and validate on the number of epochs
    for e in range(1, config.NUM_EPOCHS + 1):
        # Record epoch time
        start_time = time.time()

        # Shuffle before training
        utils.shuffle_together(X_train, y_train)

        train_loss, train_acc = go_through_by_batch(model, X_train, y_train, training=True)
        val_loss, val_acc = go_through_by_batch(model, X_val, y_val, training=False)

        end_time = time.time()

        print("Epoch {:>3}/{} | ".format(e, config.NUM_EPOCHS), end='')
        print("Train Loss: {:.4f}, Train Acc: {:.4f} | ".format(train_loss, train_acc), end='')
        if len(X_val) > 0:
            print("Val Loss: {:.4f}, Val Acc: {:.4f} | ".format(val_loss, val_acc), end='')
        print("Time spent: {:.2f} sec".format(end_time - start_time))



def model_evaluate(model, X_test, y_test):
    ''' Evaluate the model on unseen examples '''

    print('\nEvaluating model on {} unseen examples from training set...'.format(len(X_test)))
    # Evaluate model using part of unseen data from the training set
    test_loss, test_acc = go_through_by_batch(model, X_test, y_test, training=False)

    print('\nTest Loss: {:.5f} | Test Accuracy: {:.5f}\n'.format(test_loss, test_acc))



def model_predict(model, X):
    ''' Use the model to make predictions given the input data '''
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
    # List of models to compare; keep 1 model only when submitting
    models_to_test = [he_and_relu]
    model_names = ['he_and_relu']

    for model, model_name in zip(models_to_test, model_names):
        print('\nTraining model: {}'.format(model_name))

        X, y = load_training_data(config.DATA_PATH)
        y = utils.label_to_onehot_vector(y, config.NUM_CLASSES)

        # Shuffle data each time before training
        utils.shuffle_together(X, y)

        split = (config.DATA_SPLIT['train'],
                 config.DATA_SPLIT['train'] + config.DATA_SPLIT['val'],
                 config.DATA_SPLIT['train'] + config.DATA_SPLIT['val'] + config.DATA_SPLIT['test'])


        assert len(X) == split[2]    # Make sure the split sums up to number of examples

        y_train, y_val, y_test = y[:split[0]], y[split[0]:split[1]], y[split[1]:split[2]]
        X_train, X_val, X_test = X[:split[0]], X[split[0]:split[1]], X[split[1]:split[2]]

        # Preprocess: normalise after split
        X_train = utils.standardise(X_train)
        if len(X_val) > 0:
            X_val = utils.standardise(X_val)
        if len(X_test) > 0:
            X_test = utils.standardise(X_test)

        # Train and Test
        model_train(model, X_train, y_train, X_val, y_val)

        if len(X_test) > 0:
            model_evaluate(model, X_test, y_test)


        if config.PREDICTING:
            ''' Load testing data for marking '''
            X_test = load_test_data(config.DATA_PATH)
            y_test_pred = model_predict(model, X_test)
            y_test_pred = np.around(y_test_pred).astype('int64')    # Save data as int

            # Save prediction to file
            with h5py.File('{}/Predicted_labels.h5'.format(config.SAVE_PATH),'w') as H:
                H.create_dataset('label', data=y_test_pred)

            print('Predictions for {} testing data saved to "{}/Predicted_labels.h5" with dataset name "label"\n'
                .format(len(X_test), config.SAVE_PATH))



if __name__ == '__main__':
    main()

