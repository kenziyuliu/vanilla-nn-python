'''
This file stores constants and configurations
'''

''' Misc '''
# Path for training data
DATA_PATH = '../data'
# Path for saving predictions
SAVE_PATH = '../data'
# Set to true if saving predictions for `test_128.h5` to DATA_PATH
PREDICTING = True

''' Constants '''
INPUT_DIM = 128
NUM_CLASSES = 10
EPSILON = 1e-12

''' Training '''
LEARNING_RATE = 0.0005
NUM_EPOCHS = 55
BATCH_SIZE = 32
OPT = 'adam'
# Data split of train/val/test, should sum to the number of examples in the training set
DATA_SPLIT = { 'train': 60000, 'val': 0, 'test': 0 }


''' SGD '''
# Only valid if OPT == 'sgd'
NESTEROV = True


