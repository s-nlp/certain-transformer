import utils_tasks as utils
import hydra
import os

import logging

log = logging.getLogger(__name__)


def run_task(task):
    log.info(f"Task name: {task.name}")
    task_args = task.args if "args" in task else ""
    task_args = task_args.replace("$\\", "\\$")
    command = f"CUDA_VISIBLE_DEVICES={utils.WORKER_CUDA_DEVICE} HYDRA_CONFIG_PATH={task.config_path} {task.environ} python {task.command} repeat={task.repeat} {task_args}"
    log.info(f"Command: {command}")
    ret = os.system(command)
    ret = str(ret)
    log.info(f'Task "{task.name}" finished with return code: {ret}.')
    return ret


@hydra.main(config_path=os.environ["HYDRA_CONFIG_PATH"])
def main(configs):
    auto_generated_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    utils.run_tasks(configs, run_task)


if __name__ == "__main__":
    main()
