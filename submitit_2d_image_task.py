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
    parser.add_argument("--array_parallelism", type=int, default=8)
    parser.add_argument("--controller_type", type=str, default="SpatialProgressionStashed")
    parser.add_argument("--n_epochs", type=int, default=1)

    parser.add_argument("--eval", action="store_true", help="Set to evaluation mode")
    parser.add_argument("--no_pretrain", action="store_true", help="Set to pretrain mode")
    parser.add_argument("--no_mask", action="store_true", help="Set to mask mode")
    parser.add_argument("--no_retrain", action="store_true", help="Set to retrain mode")

    parser.add_argument("--non_uniform", action="store_true", help="Set to non uniform sampling")
    parser.add_argument("--folder_name", type=str, default="natural_images")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--mask_res", type=int, default=512)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    file_names = get_image_filenames(constants.DATA_ROOT / args.folder_name)
    executor = submitit.AutoExecutor(folder="logs")

    controller_type = ControllerType.__members__[args.controller_type]

    executor.update_parameters(
        timeout_min=args.timeout,
        gpus_per_node=1,
        cpus_per_task=10,
        nodes=1,
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism
    )
    pretrain = not args.no_pretrain
    learn_mask = not args.no_mask
    retrain = not args.no_retrain

    pretrain = not args.eval
    learn_mask = not args.eval
    retrain = not args.eval

    with executor.batch():
        for i in range(args.n_runs):
            for file_name in file_names:
                executor.submit(main, IMAGE_PATH=str(Path(args.folder_name) / file_name), CONTROLLER_TYPE=controller_type, EPOCHS=args.n_epochs, PRETRAIN=pretrain, LEARN_MASK=learn_mask, RETRAIN=retrain, NON_UNIFORM = args.non_uniform, MASK_RES=args.mask_res)
