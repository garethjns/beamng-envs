from gym.spaces import Box, Dict, MultiBinary, Text

BATTERY_TEST_PARAM_SPACE = dict(
    map=dict(
        values=[
            "italy_castle_town",
            "italy_village_mountain",
            "italy_crossroads",
            "italy_city",
            "italy_town_east",
            "italy_small_village",
        ],
        name="map",
        description="map",
        type=str,
        default="italy_crossroads",
    ),
    tire_pressure_f=dict(
        values=(0, 50),
        name="$tirepressure_F",
        description="Front tire pressure, psi",
        type=float,
        default=28.0,
    ),
    tire_pressure_r=dict(
        values=(0, 50),
        name="$tirepressure_R",
        description="Rear tire pressure, psi",
        type=float,
        default=27.06,
    ),
    driver_aggression=dict(
        values=(0.25, 0.65),
        name="driver_aggression",
        description="Aggressiveness setting of AI driver",
        type=float,
        default=0.5,
    ),
)

_categoricals = ["map"]
_floats = [c for c in BATTERY_TEST_PARAM_SPACE.keys() if c not in _categoricals]

_spaces = {
    v["name"]: Box(low=v["values"][0], high=v["values"][1])
    for k, v in BATTERY_TEST_PARAM_SPACE.items()
    if k in _floats
}
_spaces.update(
    {
        v["name"]: Text(min_length=1, max_length=1, charset=v["values"])
        for k, v in BATTERY_TEST_PARAM_SPACE.items()
        if k in _categoricals
    }
)

BATTERY_TEST_PARAM_SPACE_GYM = Dict(spaces=_spaces)
