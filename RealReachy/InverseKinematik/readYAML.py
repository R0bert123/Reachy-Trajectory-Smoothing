import time
import numpy as np
import yaml
from time import sleep
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

reachy = ReachySDK(host='localhost')
reachy.turn_on('reachy')

with open("/home/reachy/Codes/Robert/YAML-Files/move_leftArm.yaml") as f:
    data = yaml.safe_load(f)

for step in data["trajectory"]:
    goto(
        {getattr(reachy.l_arm, j): v for j, v in step["joints"].items()},
        duration=step["duration"],
        interpolation_mode=InterpolationMode.MINIMUM_JERK
    )

reachy.turn_off_smoothly('reachy')

