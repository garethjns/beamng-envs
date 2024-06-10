"""
Run the DragStrip environment a number of times, load and compare results.
"""
import os

import mlflow
from beamngpy import BeamNGpy
from matplotlib import pyplot as plt
from tqdm import tqdm

from beamng_envs import __VERSION__
from beamng_envs.bng_sim.beamngpy_config import BeamNGPyConfig

from beamng_envs.envs.drag_strip.drag_strip_config import DragStripConfig
from beamng_envs.envs.drag_strip.drag_strip_env import DragStripEnv
from scripts.args_batch import PARSER_BATCH
from scripts.run_single_drag_strip import plot_drag_strip

if __name__ == "__main__":
    opt = PARSER_BATCH.parse_args()

    config = DragStripConfig(
        bng_config=BeamNGPyConfig(
            home=opt.beamng_path,
            user=opt.beamng_user_path,
        ),
        fps=20,
        max_time=30,
        close_on_done=False,
        error_on_out_of_time=False,
    )
    bng = BeamNGpy(**config.bng_config.__dict__)

    # Set up the mlflow experiment
    mlflow.set_experiment("Drag strip example")

    for run in tqdm(range(opt.N)):
        params = DragStripEnv.param_space.sample()

        with mlflow.start_run():
            env = DragStripEnv(params=params, config=config, bng=bng)
            results, _ = env.run()

            plot_drag_strip(
                env_history=env.history,
                label=f"run {run}",
                filename=os.path.join(
                    env.disk_results.output_path,
                    "drag_strip_plot.png"),
            )

            # Log to MLflow
            params.update({"version": __VERSION__})
            # Note here params contains the whole requested part config... This is a lot, it may break
            # MLflow...
            mlflow.log_params(params)
            _ = results.pop("parts_requested")
            _ = results.pop("parts_actual")
            mlflow.log_metrics({k: float(v) for k, v in results.items()})
            mlflow.log_artifacts(env.disk_results.output_path)
