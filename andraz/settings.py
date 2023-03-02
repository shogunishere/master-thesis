try:
    from andraz.local_settings import *
except:
    pass

PROJECT_DIR = ""
SEED = 4231
BATCH_SIZE = 2
# EPOCHS = 1000
EPOCHS = 10
LEARNING_RATE = 0.0001
CLASSES = [0, 1, 3, 4]
WANDB = False
WIDTHS = [0.25, 0.50, 0.75, 1.0]


try:
    from andraz.local_settings import *
except:
    pass
