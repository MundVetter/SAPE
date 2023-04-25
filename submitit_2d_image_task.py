import submitit
from tasks_image_2d import main
import constants
import argparse
import os
from pathlib import Path
from custom_types import *

def get_image_filenames(folder_path):
    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter the file list to include only images with common extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_filenames = [file for file in file_list if any(file.lower().endswith(ext) for ext in image_extensions)]

    return image_filenames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--array_parallelism", type=int, default=3)
    parser.add_argument("--controller_type", type=str, default="GlobalProgression")
    parser.add_argument("--n_epochs", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    file_names = get_image_filenames(constants.DATA_ROOT / "natural_images")
    executor = submitit.AutoExecutor(folder="logs")

    controller_type = ControllerType.__members__[args.controller_type]

    executor.update_parameters(
        timeout_min=20,
        gpus_per_node=1,
        cpus_per_task=10,
        nodes=1,
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism
    )

    with executor.batch():
        for i in range(args.n_runs):
            for file_name in file_names:
                executor.submit(main, IMAGE_PATH = str(Path("natural_images") / file_name), CONTROLLER_TYPE = controller_type, EPOCHS = args.n_epochs)
