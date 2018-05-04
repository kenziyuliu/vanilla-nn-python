import config
from initializers import xavier_uniform_init, he_uniform_init
from layers import FullyConnected, LeakyReLU, Dropout, SoftmaxCrossEntropy

'''
Model and its output activation and loss function
Currently the output gate assumes Softmax + CrossEntropy
'''
good_model = [
    FullyConnected(config.INPUT_DIM, 256, xavier_uniform_init, use_weight_norm=True, use_bias=True),
    LeakyReLU(),
    Dropout(0.5),
    FullyConnected(256, 64, xavier_uniform_init, use_weight_norm=True, use_bias=True),
    LeakyReLU(),
    Dropout(0.3),
    FullyConnected(64, 32, xavier_uniform_init, use_weight_norm=True, use_bias=True),
    LeakyReLU(),
    Dropout(0.1),
    FullyConnected(32, config.NUM_CLASSES, xavier_uniform_init, use_weight_norm=True, use_bias=True)
]

softmax_crossentropy = SoftmaxCrossEntropy()