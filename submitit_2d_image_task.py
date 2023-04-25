import submitit
from tasks_image_2d import main
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, default="gpu_shared")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--array_parallelism", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        timeout_min=20,
        gpus_per_node=1,
        cpus_per_task=10,
        nodes=1,
        slurm_signal_delay_s=120
    )