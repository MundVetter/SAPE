import submitit
import tasks_image_2d
import tasks_occupancy_3d
import constants
import argparse
from pathlib import Path
from custom_types import *
import uuid
from utils.files_utils import get_image_filenames, get_mesh_filenames

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--array_parallelism", type=int, default=8)
    parser.add_argument("--controller_type", type=str, default="SpatialProgressionStashed")
    parser.add_argument("--encoding_type", type=str, default="FF")
    parser.add_argument("--n_epochs", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--non_uniform", action="store_true", help="Set to non uniform sampling")
    parser.add_argument("--folder_name", type=str, default="natural_images")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--mask_res", type=int, default=512)
    parser.add_argument("--no_inv_prob", action='store_true')
    parser.add_argument('--weight_decay', type=int, default=1)

    parser.add_argument("--lambda_cost", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1 - 1e-3)
    parser.add_argument("--sigma", type=float, default=20.)


    parser.add_argument("--render_res", type=int, default=256)
    parser.add_argument("--task", type=str, default="2d")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.task == "2d":
        file_names = get_image_filenames(constants.DATA_ROOT / args.folder_name)
        main = tasks_image_2d.main
    elif args.task == "3d":
        file_names = get_mesh_filenames(constants.DATA_ROOT / args.folder_name)
        main = tasks_occupancy_3d.main
    else:
        raise ValueError("Task should be either 2d or 3d")

    executor = submitit.AutoExecutor(folder="logs")

    controller_type = ControllerType.__members__[args.controller_type]
    encoding_type = EncodingType.__members__[args.encoding_type]

    executor.update_parameters(
        timeout_min=args.timeout,
        gpus_per_node=1,
        cpus_per_task=10,
        nodes=1,
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism
    )

    with executor.batch():
        for i in range(args.n_runs):
            group_name = str(uuid.uuid4())[:8]
            for file_name in file_names:
                executor.submit(main, PATH=str(Path(args.folder_name) / file_name), CONTROLLER_TYPE=controller_type, EPOCHS=args.n_epochs, NON_UNIFORM = args.non_uniform, MASK_RES=args.mask_res, LAMBDA_COST=args.lambda_cost, RUN_NAME=group_name, THRESHOLD=args.threshold, SIGMA=args.sigma, BATCH_SIZE=args.batch_size, LR=args.lr, ENCODING_TYPE=encoding_type, RENDER_RES=args.render_res, INV_PROB = not args.no_inv_prob, WEIGHT_DECAY = args.weight_decay)

