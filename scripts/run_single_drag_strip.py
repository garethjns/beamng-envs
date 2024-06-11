"""
Run the DragStrip environment a number of times, load and compare results.
"""

import os
from typing import Any, Dict, Optional

import mlflow
from beamngpy import BeamNGpy
from matplotlib import pyplot as plt
from tqdm import tqdm

from beamng_envs import __VERSION__
from beamng_envs.bng_sim.beamngpy_config import BeamNGPyConfig
from beamng_envs.envs.drag_strip.drag_strip_config import DragStripConfig
from beamng_envs.envs.drag_strip.drag_strip_env import DragStripEnv
from beamng_envs.envs.history import History
from scripts.args_batch import PARSER_BATCH


def plot_drag_strip(
    env_history: History,
    label: str = "",
    filename: Optional[str] = None,
):
    fig, ax = plt.subplots()

    ax.plot(
        env_history["time_s"],
        [t["state"]["pos"][0] for t in env_history["car_state"]],
        label=label,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance travelled")
    ax.legend()

    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    opt = PARSER_BATCH.parse_args()

    config = DragStripConfig(
        bng_config=BeamNGPyConfig(
            home=opt.beamng_path,
            user=opt.beamng_user_path,
        ),
        fps=20,
        max_time=30,
        close_on_done=True,
        error_on_out_of_time=False,
    )
    bng = BeamNGpy(**config.bng_config.__dict__)

    env = DragStripEnv(params=DragStripEnv.param_space.sample(), config=config, bng=bng)
    results, _ = env.run()

    plot_drag_strip(
        env_history=env.history,
        label="single run",
    )
