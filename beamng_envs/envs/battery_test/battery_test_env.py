from typing import Any, Dict, Iterable, Optional, Tuple

from beamngpy import BeamNGpy

from beamng_envs.bng_sim.bng_sim import BNGSim
from beamng_envs.data.disk_results import DiskResults
from beamng_envs.envs.battery_test.battery_test_config import BatteryTestConfig
from beamng_envs.envs.battery_test.battery_test_paradigm import BatteryTestParadigm
from beamng_envs.envs.battery_test.battery_test_param_space import (
    BATTERY_TEST_PARAM_SPACE_GYM,
)
from beamng_envs.envs.history import History
from beamng_envs.envs.track_test.track_test_paradigm import TrackTestParadigm
from beamng_envs.interfaces.env import IEnv


class BatteryTestEnv(IEnv):
    """
    Prototype battery test environment,

    Loads the eSBR 800 electric car and drives as traffic for a set period of time.

    Car config is customisable and car data, including battery level is recorded

    See scripts/ for some usage examples.
    """

    param_space = BATTERY_TEST_PARAM_SPACE_GYM

    def __init__(
        self,
        params: Dict[str, Any],
        config: BatteryTestConfig = BatteryTestConfig(),
        bng: Optional[BeamNGpy] = None,
    ):
        self.params = params
        self.config = config
        self.history = History()
        self.disk_results = None

        self._bng_simulation = BNGSim(config=config, bng=bng)
        self._paradigm = BatteryTestParadigm(params=params)

    def step(
        self, action: Optional[int] = None, **kwargs
    ) -> Tuple[Optional[Any], Optional[float], bool, Dict[str, Any]]:
        """

        :param action:
        :param kwargs:
        :return: Tuple containing (observation, reward, done, info) (to match OpenAI Gym interface).
        """
        return self._paradigm.step(action=action, bng_simulation=self._bng_simulation)

    def run(
        self, modifiers: Optional[Dict[str, Iterable[Any]]] = None
    ) -> Tuple[Dict[str, Any], History]:
        self.reset()
        current_time_s = 0

        if self.done:
            raise ValueError("Finished, reset before use.")

        while not self.done:
            obs, _, done, _ = self.step(bng_simulation=self._bng_simulation)
            current_time_s = self._bng_simulation.get_real_time(
                self._paradigm.current_step
            )
            self.done = self._paradigm.done

            self.history.append(
                {
                    self.history.step_key: self._paradigm.current_step,
                    self.history.time_key: current_time_s,
                    self.history.car_state_key: obs,
                }
            )

        bng_logs_path = self._bng_simulation.stop_bng_logging_for(
            self._paradigm.vehicle
        )
        self.results = {
            self.history.time_key: current_time_s,
            "finished": self._paradigm.finished,
        }

        self.disk_results = DiskResults(
            path=self.config.output_path,
            params=self.params,
            config=self.config.__dict__,
            history=self.history.__dict__,
            path_to_bng_logs=bng_logs_path,
            results=self.results,
        )
        self.disk_results.save()
        self._bng_simulation.close()

        return self.results, self.history

    def reset(self) -> None:
        self.done = False
        self._bng_simulation.reset()
        self._paradigm.reset(bng_simulation=self._bng_simulation)
        self.history.reset()
        self.results = {}
