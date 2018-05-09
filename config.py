'''
This file stores constants and configurations
'''

''' Misc '''
DATA_PATH = '../data'       # For loading data
PREDICTING = False          # Set to true if saving predictions for `test_128.h5` to DATA_PATH

''' Constants '''
INPUT_DIM = 128
NUM_CLASSES = 10
EPSILON = 1e-12

''' Training '''
LEARNING_RATE = 0.0005
NUM_EPOCHS = 65
BATCH_SIZE = 32
OPT = 'adam'
# Data split of train/val/test, should sum to the number of examples in the training set
DATA_SPLIT = { 'train': 58000, 'val': 1000, 'test': 1000 }


''' SGD '''     # Only if OPT == 'sgd'
NESTEROV = True


