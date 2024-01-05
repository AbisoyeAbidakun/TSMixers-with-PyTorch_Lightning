# Training hyperparameters
INPUT_SIZE = 784
#Input  Sequence Lenght 
SEQ_LEN =  336
# prediction length 
PRED_LEN = 96 
# learning rate
LEARNING_RATE = 0.001

BATCH_SIZE = 32 
NUM_EPOCHS = 100

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "mps"
DEVICES = -1
PRECISION = '16-mixed'


