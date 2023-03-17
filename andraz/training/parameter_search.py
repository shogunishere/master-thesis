from threading import Thread
from time import sleep

from andraz.training.training import Training

# DEVICES = ["cuda:0", "cuda:1", "cuda:2"]
DEVICES = ["cuda:0"]
RUNS_PER_GROUP = 1

BATCH_SIZE = 2**7
EPOCHS = 1000
LEARNING_RATE = 0.0001
REGULARISATION_L2 = 0.1
IMAGE_RESOLUTION = (128, 128)

# List of dicts with parameters to explore.
PARAMETERS = [
    {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "LEARNING_RATE_SCHEDULER": "linear",
        "REGULARISATION_L2": REGULARISATION_L2,
        "IMAGE_RESOLUTION": IMAGE_RESOLUTION,
    },
    {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "LEARNING_RATE_SCHEDULER": "no scheduler",
        "REGULARISATION_L2": REGULARISATION_L2,
        "IMAGE_RESOLUTION": IMAGE_RESOLUTION,
    },
    {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "LEARNING_RATE_SCHEDULER": "exponential",
        "REGULARISATION_L2": REGULARISATION_L2,
        "IMAGE_RESOLUTION": IMAGE_RESOLUTION,
    },
]


class GPUThread(Thread):
    def __init__(self, device, parameters):
        super().__init__()
        self.device = device
        self.parameters = parameters

        self.done = False

    def run(self):
        for _ in range(RUNS_PER_GROUP):
            tr = Training(
                device=self.device,
                epochs=self.parameters["EPOCHS"],
                learning_rate=self.parameters["LEARNING_RATE"],
                learning_rate_scheduler=self.parameters["LEARNING_RATE_SCHEDULER"],
                batch_size=self.parameters["BATCH_SIZE"],
                regularisation_l2=self.parameters["REGULARISATION_L2"],
                image_resolution=self.parameters["IMAGE_RESOLUTION"],
                verbose=0,
                wandb_group=self.parameters["LEARNING_RATE_SCHEDULER"] + " 003",
            )
            tr.train()
        self.done = True


if __name__ == "__main__":
    threads = []
    param_idx = 0

    # Start one thread on each device
    for device in DEVICES:
        thread = GPUThread(device, PARAMETERS[param_idx])
        thread.start()
        threads.append(thread)
        param_idx += 1

    # Check if the device is done and run with new parameters
    report = 0
    while len(threads) > 0:
        for thread in threads:
            if thread.done:
                thread.join()
                device = thread.device
                threads.remove(thread)
                if param_idx < len(PARAMETERS):
                    thread = GPUThread(device, PARAMETERS[param_idx])
                    thread.start()
                    threads.append(thread)
                    param_idx += 1

        # Report current state
        if report > 6 * 5:
            print("========================================================")
            print("Param index: {}/{}".format(param_idx, len(PARAMETERS)))
            print("Active threads: {}".format([x.device for x in threads]))
            print("========================================================")
            print()
            report = 0

        sleep(10)
