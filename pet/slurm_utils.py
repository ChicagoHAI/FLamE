import os
from subprocess import call
import signal
import threading
import time
import logging

logger = logging.getLogger(__name__)

def is_on_slurm():
    return os.environ.get("SLURM_JOB_ID") is not None

def schedule_death(seconds, verbose=False):
    logger.info(f"scheduling death after {seconds}s")
    def f():
        death = time.time() + seconds
        while time.time() < death:
            if verbose:
                logger.info(f"Beep...")
            sleep_interval = max(0, min(600, death - time.time()))
            time.sleep(sleep_interval)
        
        logger.info(f"time to die...")
        logging.shutdown()
        os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=f, daemon=True).start()

def slurm_sigusr1_handler_fn(signum, frame) -> None:
    logger.info(f"received signal {signum}")
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = ["scontrol", "requeue", job_id]

    logger.info(f"requeing job {job_id}...")
    try:
        result = call(cmd)
    except FileNotFoundError:
        joint_cmd = [str(x) for x in cmd]
        result = call(" ".join(joint_cmd), shell=True)

    if result == 0:
        logger.info(f"requeued exp {job_id}")
    else:
        logger.info("requeue failed")

def setup_slurm():
    if not is_on_slurm():
        logger.info("not running in slurm, this job will run until it finishes.")
        return
    logger.info("running in slurm, ready to requeue on SIGUSR1.")
    signal.signal(signal.SIGUSR1, slurm_sigusr1_handler_fn)
    # slurm not sending the signal, so sending it myself
    time_to_live = 14300 # just a bit less than 4 hrs
    schedule_death(time_to_live)
