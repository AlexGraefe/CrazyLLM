import os
import logging
from functools import wraps


def world_info_from_env():
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def initialize_logger():
    """Initializes and returns a logger."""

    _, global_rank, world_size = world_info_from_env()

    logger = logging.getLogger("execution_guard")
    logger.handlers.clear()  # Clear existing handlers to avoid duplicates

    formatter = logging.Formatter(
        f"[%(asctime)s] [{global_rank}] %(levelname)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(f'execution_guard_{global_rank}.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # if global_rank == 0:
    #     # Add a console handler for the main process
    #     ch = logging.StreamHandler()
    #     ch.setLevel(logging.DEBUG)
    #     ch.setFormatter(formatter)
    #     logger.addHandler(ch)

    logger.setLevel(logging.DEBUG)
    return logger

def execution_guard(func, force_overwrite=False):
    """Decorator to execute the function only if 'finished.lock' is absent in the current directory."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        finished_lock = os.path.join(os.getcwd(), "finished.lock")
        if os.path.exists(finished_lock) and not force_overwrite:
            logger = initialize_logger()
            logger.info("Operation aborted. 'finished.lock' file found in the current directory.")
            return None
        else:
            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".log"):
                    try:
                        os.remove(filename)
                        print(f"Deleted log file: {filename}")
                    except Exception as e:
                        print(f"Failed to delete log file: {filename}")
            logger = initialize_logger()
        try:
            logger.info(f"Executing function: {func.__name__}")
            func(*args, **kwargs)
            with open(finished_lock, 'w') as lock_file:
                lock_file.write("finished")
        except Exception as e:
            logger.exception("Exception occurred during function execution.")
    return wrapper

# Example usage:
#
# @execution_guard
# def function_name(logger, *args, **kwargs):
#     logger.info("Function is running with args: %s, kwargs: %s", args, kwargs)
#
# function_name("example_arg", key="value")