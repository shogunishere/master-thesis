try:
    from andraz.local_settings import *
except:
    pass

PROJECT_DIR = ""
SEED = 4231

BATCH_SIZE = 2**8
# BATCH_SIZE = 2**4
EPOCHS = 1000
# EPOCHS = 150

LEARNING_RATE = 0.001
# LEARNING_RATE_SCHEDULER = "linear"
LEARNING_RATE_SCHEDULER = "exponential"
# LEARNING_RATE_SCHEDULER = "no scheduler"

# MODEL = "slim"
MODEL = "squeeze"

REGULARISATION_L2 = 0.1

DROPOUT = 0.9

REDUCE_RESOLUTION = True
# REDUCE_RESOLUTION = False
IMAGE_RESOLUTION = (128, 128)

CLASSES = [0, 1, 3, 4]
WANDB = False
WIDTHS = [0.25, 0.50, 0.75, 1.0]


try:
    from andraz.local_settings import *
except:
    pass
