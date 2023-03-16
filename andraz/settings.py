try:
    from andraz.local_settings import *
except:
    pass

PROJECT_DIR = ""
SEED = 4231
BATCH_SIZE = 64
EPOCHS = 1000
# EPOCHS = 100
LEARNING_RATE = 0.00001
REGULARISATION_L2 = 0.1
CLASSES = [0, 1, 3, 4]
WANDB = False
WIDTHS = [0.25, 0.50, 0.75, 1.0]


try:
    from andraz.local_settings import *
except:
    pass
