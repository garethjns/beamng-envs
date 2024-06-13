"""
Runs a single track test, plots results.

For running batches of experiments, see scripts/run_batch_track_tests.py

To run, install requirements and call specifying beamng_path options (see readme)

````
pip install -r scripts/requirements.txt
python -m scripts.run_single_track_test --beamng_path '' --beamng_user_path ''
````

"""

import beamngpy
import numpy as np

from beamng_envs.bng_sim.beamngpy_config import BeamNGPyConfig
from beamng_envs.data.disk_results import DiskResults
from beamng_envs.envs.track_test.track_test_config import TrackTestConfig
from beamng_envs.envs.track_test.track_test_env import TrackTestEnv
from beamng_envs.envs.track_test.track_test_param_space import (
    TRACK_TEST_PARAM_SPACE_GYM,
)
from scripts.args_single import PARSER_SINGLE

if __name__ == "__main__":
    opt = PARSER_SINGLE.parse_args()

    # Setup Python logging to include BeamNG console output
    beamngpy.set_up_simple_logging()

    # Prepare config
    # The BeamNGConfig can be specified here, and will be used to create a game instance if the env isn't passed an
    # existing one.
    track_test_config = TrackTestConfig(
        close_on_done=True,
        output_path=opt.output_path,
        bng_config=BeamNGPyConfig(home=opt.beamng_path, user=opt.beamng_user_path),
    )

    # Sample a single set of parameters
    p_set = TRACK_TEST_PARAM_SPACE_GYM.sample()

    # Run the test for each set of parameters
    env = TrackTestEnv(params=p_set, config=track_test_config)

    # Run the env
    results, history = env.run()

    # Previous results can also be reloaded from disk (these are also available in current runs in env.disk_results)
    full_results = DiskResults.load(path=env.disk_results.output_path)

    # Get the tabulated results
    timeseries_df = full_results.ts_df

    full_results.plot_track(
        row_2_series=["electrics_throttle_0", "electrics_brake_0", "electrics_gear_0"],
        row_3_series=["damage_damage_0"],
    )
    print(f"Lap time: {np.round(results['time_s'],2)}s")
