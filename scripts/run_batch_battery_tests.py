"""Run a number of samples of car configs in the BatteryTestEnv, with logging to MLflow."""

import os

import mlflow
from beamngpy import BeamNGpy
from tqdm import tqdm

from beamng_envs import __VERSION__
from beamng_envs.bng_sim.beamngpy_config import BeamNGPyConfig
from beamng_envs.envs.battery_test.battery_test_config import BatteryTestConfig
from beamng_envs.envs.battery_test.battery_test_env import BatteryTestEnv
from beamng_envs.envs.battery_test.battery_test_param_space import (
    BATTERY_TEST_PARAM_SPACE_GYM,
)
from scripts.args_batch import PARSER_BATCH

if __name__ == "__main__":
    opt = PARSER_BATCH.parse_args()

    # Prepare config
    battery_test_config = BatteryTestConfig(
        close_on_done=True,
        max_time=60,
        output_path=opt.output_path,
        bng_config=BeamNGPyConfig(home=opt.beamng_path, user=opt.beamng_user_path),
    )

    # Sample N sets of parameters
    param_sets = [BATTERY_TEST_PARAM_SPACE_GYM.sample() for _ in range(opt.N)]

    # Set up the mlflow experiment
    mlflow.set_experiment("Battery test example")

    # Create a persistent game instance to use to run multiple tests in
    with BeamNGpy(**battery_test_config.bng_config.__dict__) as bng:
        # Run the test for each set of parameters
        for p_set in tqdm(param_sets):
            env = BatteryTestEnv(
                params=p_set,
                config=battery_test_config,
                bng=bng,
            )

            with mlflow.start_run():
                # Run the env
                results, history = env.run()

                # Collect params and metrics, log to MLflow
                params = {k.replace("$", ""): float(v) for k, v in p_set.items()}
                params.update({"version": __VERSION__})
                mlflow.log_params(params)
                results["finished"] = float(results["finished"])
                mlflow.log_metrics(results)

                # Add a plot of the track
                env.disk_results.plot_track(
                    row_2_series=[
                        "electrics_throttle_input_0",
                        "electrics_brake_0",
                    ],
                    row_3_series=["electrics_fuel_0"],
                    filename=os.path.join(
                        env.disk_results.output_path, "track_plot.png"
                    ),
                )
                mlflow.log_artifacts(env.disk_results.output_path)
