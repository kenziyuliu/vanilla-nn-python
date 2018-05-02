'''
This file stores constants and configurations
'''

''' Constants '''
INPUT_DIM = 128
NUM_CLASSES = 10
EPSILON = 1e-12

''' Training '''
LEARNING_RATE = 0.001
OPT='sgd'
NUM_EPOCHS = 1
BATCH_SIZE = 64

# from layers import *
#
# # Example usage
# sample_model = [
#     FullyConnected(INPUT_DIM, 256, weight_initializer=None),
#     LeakyReLU(alpha=0.1),
#     FullyConnected(256, 128, weight_initializer=None),
#     LeakyReLU(),
#     FullyConnected(128, 64, weight_initializer=None),
#     ReLU(),
#     FullyConnected(64, 10, weight_initializer=None),
#     Softmax()
# ]
#
# # Example
# data = np.random.randint(1, 100, size=128)
# for layer in sample_model:
#     data = layer.forward(data)
#
# y_true = np.zeros(10)
# y_true[1] = 1
#
# lossfunction = CrossEntropyLoss()
# loss = lossfunction.forward(data, y_true)
# gradient = lossfunction.backward()
#
# for layer in reversed(sample_model):
#     gradient = layer.backward(gradient)
