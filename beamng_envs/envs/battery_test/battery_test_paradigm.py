from typing import Any, Dict, Optional, Tuple

from beamngpy import Scenario, Vehicle
from beamngpy.types import Float3

from beamng_envs.bng_sim.bng_sim import BNGSim
from beamng_envs.cars.e_sbr_800 import ESBR800
from beamng_envs.interfaces.paradigm import IParadigm
from beamng_envs.locations import italy
from beamng_envs.locations.spawn_location import SpawnLocation
from beamng_envs.maths import euclidean_distance_3d


class BatteryTestParadigm(IParadigm):
    _car_model = "sbr"
    _car_default_config = ESBR800()
    _car_spawn_locations: Dict[str, SpawnLocation] = dict(
        italy_castle_town=italy.ItalyCastleTown,
        italy_village_mountain=italy.ItalyVillageMountain,
        italy_crossroads=italy.ItalyCrossroads,
        italy_city=italy.ItalyCity,
        italy_town_east=italy.ItalyTownEast,
        italy_small_village=italy.ItalySmallVillage,
    )
    _last_step_location: Float3
    _total_distance_traveled: float
    done: bool
    current_step: int
    vehicle: Vehicle

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self._ready = False

    @property
    def _car_spawn_pos(self) -> SpawnLocation:
        return self._car_spawn_locations[self.params["map"]]

    @property
    def _map_name(self) -> str:
        return self._car_spawn_locations[self.params["map"]].map

    def start_scenario(self, bng_simulation: BNGSim):
        self._scenario = Scenario(level=self._map_name, name="battery_test")
        self.vehicle = Vehicle(
            self._car_model,
            model=self._car_model,
            licence="MONOLITH",
            part_config="electric_800.pc",
            color="blue",
        )
        self._scenario.add_vehicle(
            vehicle=self.vehicle,
            pos=self._car_spawn_pos.pos,
            rot_quat=self._car_spawn_pos.rot_quat,
        )
        self._scenario.make(bng_simulation.bng)
        bng_simulation.start_scenario(self._scenario)

        pc = self._car_default_config.config.copy()
        pc["vars"].update(
            {k: float(v) for k, v in self.params.items() if k.startswith("$")}
        )

        self.vehicle.set_part_config(pc)
        self.vehicle.disconnect()
        self.vehicle.connect(bng=bng_simulation.bng)
        bng_simulation.attach_sensors_to_vehicle(self.vehicle)
        bng_simulation.bng.switch_vehicle(self.vehicle)
        self.vehicle.ai_set_mode("traffic")
        self.vehicle.ai_set_aggression(float(self.params["driver_aggression"]))

    def step(
        self, bng_simulation: BNGSim, action: Optional[int] = None, **kwargs
    ) -> Tuple[Optional[Any], Optional[float], bool, Dict[str, Any]]:
        if not self._ready:
            self.reset(bng_simulation)

        if self.done:
            raise ValueError("Finished")

        bng_simulation.bng.step(1)
        sensor_data = bng_simulation.poll_sensors_for_vehicle(self.vehicle)

        self.done = bng_simulation.check_time_limit(self.current_step)
        self.current_step += 1

        dist = euclidean_distance_3d(
            pos_1=self.vehicle.state["pos"], pos_2=self._last_step_location
        )
        self._total_distance_traveled += dist
        sensor_data["distance_traveled_step"] = dist
        sensor_data["distance_traveled_total"] = self._total_distance_traveled
        self._last_step_location = self.vehicle.state["pos"]

        return sensor_data, None, self.done, {}

    def reset(self, bng_simulation: BNGSim):
        self._ready = True
        self.start_scenario(bng_simulation)
        self.done = False
        self.finished = False
        self.current_step = 0
        self._last_step_location = self._car_spawn_locations[self.params["map"]].pos
        self._total_distance_traveled = 0.0
