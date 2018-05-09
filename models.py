import config
from initializers import xavier_uniform_init, he_uniform_init, he_normal_init
from layers import FullyConnected
from layers import LeakyReLU, ReLU, Sigmoid, Tanh, SoftmaxCrossEntropy
from layers import BatchNorm, Dropout

'''
Model and its output activation and loss function
Currently the output gate assumes Softmax + CrossEntropy
'''

softmax_crossentropy = SoftmaxCrossEntropy()

he_and_relu = [     # CURRENTLY THE BEST MODEL
    FullyConnected(config.INPUT_DIM, 192, he_uniform_init, use_weight_norm=True, weight_decay=3e-7),
    BatchNorm(input_dim=192),
    ReLU(),
    Dropout(0.4),

    FullyConnected(192, 96, he_uniform_init, use_weight_norm=True, weight_decay=3e-7),
    BatchNorm(input_dim=96),
    ReLU(),
    Dropout(0.4),

    FullyConnected(96, 48, he_uniform_init, use_weight_norm=True, weight_decay=3e-7),
    BatchNorm(input_dim=48),
    ReLU(),
    Dropout(0.3),

    FullyConnected(48, config.NUM_CLASSES, he_uniform_init, use_weight_norm=True),
]

xavier_and_lrelu = [
    FullyConnected(config.INPUT_DIM, 192, xavier_uniform_init, use_weight_norm=True, weight_decay=1e-6),
    BatchNorm(input_dim=192),
    LeakyReLU(),
    Dropout(0.4),

    FullyConnected(192, 96, xavier_uniform_init, use_weight_norm=True, weight_decay=1e-6),
    BatchNorm(input_dim=96),
    LeakyReLU(),
    Dropout(0.4),

    FullyConnected(96, 48, xavier_uniform_init, use_weight_norm=True),
    BatchNorm(input_dim=48),
    LeakyReLU(),
    Dropout(0.4),

    FullyConnected(48, config.NUM_CLASSES, xavier_uniform_init, use_weight_norm=True),
]


relu_with_sgd = [
    FullyConnected(config.INPUT_DIM, 256, xavier_uniform_init, opt='sgd', use_weight_norm=True, use_bias=False),
    Dropout(0.4),
    BatchNorm(input_dim=256, opt='sgd'),
    ReLU(),

    FullyConnected(256, 64, xavier_uniform_init, opt='sgd', use_weight_norm=True, use_bias=False),
    Dropout(0.4),
    BatchNorm(input_dim=64, opt='sgd'),
    ReLU(),

    FullyConnected(64, 32, xavier_uniform_init, opt='sgd', use_weight_norm=True, use_bias=False),
    Dropout(0.1),
    BatchNorm(input_dim=32, opt='sgd'),
    ReLU(),

    FullyConnected(32, config.NUM_CLASSES, xavier_uniform_init, opt='sgd', use_weight_norm=True, use_bias=False)
]


sigmoid_model = [
    FullyConnected(config.INPUT_DIM, 256, xavier_uniform_init, use_weight_norm=True, use_bias=False),
    Dropout(0.4),
    BatchNorm(input_dim=256),
    Sigmoid(),

    FullyConnected(256, 64, xavier_uniform_init, use_weight_norm=True, use_bias=False),
    Dropout(0.4),
    BatchNorm(input_dim=64),
    Sigmoid(),

    FullyConnected(64, 32, xavier_uniform_init, use_weight_norm=True, use_bias=False),
    Dropout(0.1),
    BatchNorm(input_dim=32),
    Sigmoid(),

    FullyConnected(32, config.NUM_CLASSES, xavier_uniform_init, use_weight_norm=True, use_bias=False)
]

tanh_model = [
    FullyConnected(config.INPUT_DIM, 256, xavier_uniform_init, use_weight_norm=True, use_bias=False),
    Dropout(0.4),
    BatchNorm(input_dim=256),
    Tanh(),

    FullyConnected(256, 64, xavier_uniform_init, use_weight_norm=True, use_bias=False),
    Dropout(0.4),
    BatchNorm(input_dim=64),
    Tanh(),

    FullyConnected(64, 32, xavier_uniform_init, use_weight_norm=True, use_bias=False),
    Dropout(0.1),
    BatchNorm(input_dim=32),
    Tanh(),

    FullyConnected(32, config.NUM_CLASSES, xavier_uniform_init, use_weight_norm=True, use_bias=False)
]
