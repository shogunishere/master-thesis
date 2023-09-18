try:
    from andraz.local_settings import *
except:
    pass

PROJECT_DIR = ""
# SEED = 4231
SEED = 123

# BATCH_SIZE = 2**8
# BATCH_SIZE = 2**4
# BATCH_SIZE = 2**6
# BATCH_SIZE = 2**1
BATCH_SIZE = 2**2
EPOCHS = 1000
# EPOCHS = 150

LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.001
# LEARNING_RATE_SCHEDULER = "linear"
LEARNING_RATE_SCHEDULER = "exponential"
# LEARNING_RATE_SCHEDULER = "no scheduler"

# MODEL = "slim"
MODEL = "squeeze"

REGULARISATION_L2 = 0.1

DROPOUT = 0.5
# DROPOUT = 0.75
# DROPOUT = False

# IMAGE_RESOLUTION = None
# IMAGE_RESOLUTION = (128, 128)
# IMAGE_RESOLUTION = (256, 256)
IMAGE_RESOLUTION = (512, 512)

CLASSES = [0, 1, 3, 4]
WANDB = False
WIDTHS = [0.25, 0.50, 0.75, 1.0]

KNN_WIDTHS = {
    0.25: 15,
    0.50: 4,
    0.75: 4,
}

# Infest loss weights
# LOSS_WEIGHTS = [0.1, 0.45, 0.45]
# Cofly loss weights
LOSS_WEIGHTS = [0.1, 0.9]


try:
    from andraz.local_settings import *
except:
    pass
