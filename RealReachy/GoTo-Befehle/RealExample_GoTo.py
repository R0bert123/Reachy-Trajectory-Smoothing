import time

import numpy as np
from time import sleep

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

reachy = ReachySDK(host='localhost')

reachy.turn_on('reachy')

poses = []

########################################################################


poses.append({
    reachy.r_arm.r_shoulder_pitch: 11.56,
    reachy.r_arm.r_shoulder_roll: -5.32,
    reachy.r_arm.r_arm_yaw: 23.91,
    reachy.r_arm.r_elbow_pitch: -111.36,
    reachy.r_arm.r_forearm_yaw: 1.67,
    reachy.r_arm.r_wrist_pitch: -4.83,
    reachy.r_arm.r_wrist_roll: 17.49,
    reachy.r_arm.r_gripper: -38.06,
})
                
poses.append({
    reachy.l_arm.l_shoulder_pitch: 15.6,
    reachy.l_arm.l_shoulder_roll: 4.44,
    reachy.l_arm.l_arm_yaw: -15.03,
    reachy.l_arm.l_elbow_pitch: -112.32,
    reachy.l_arm.l_forearm_yaw: -0.88,
    reachy.l_arm.l_wrist_pitch: -2.81,
    reachy.l_arm.l_wrist_roll: -27.95,
    reachy.l_arm.l_gripper: 54.4,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 6.55,
    reachy.r_arm.r_shoulder_roll: -12.17,
    reachy.r_arm.r_arm_yaw: 6.68,
    reachy.r_arm.r_elbow_pitch: -101.07,
    reachy.r_arm.r_forearm_yaw: -3.69,
    reachy.r_arm.r_wrist_pitch: -3.43,
    reachy.r_arm.r_wrist_roll: 15.73,
    reachy.r_arm.r_gripper: -5.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 8.04,
    reachy.r_arm.r_shoulder_roll: -10.76,
    reachy.r_arm.r_arm_yaw: -8.53,
    reachy.r_arm.r_elbow_pitch: -100.63,
    reachy.r_arm.r_forearm_yaw: 8.61,
    reachy.r_arm.r_wrist_pitch: -4.57,
    reachy.r_arm.r_wrist_roll: 32.26,
    reachy.r_arm.r_gripper: -5.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 9.8,
    reachy.r_arm.r_shoulder_roll: -10.59,
    reachy.r_arm.r_arm_yaw: -21.36,
    reachy.r_arm.r_elbow_pitch: -101.34,
    reachy.r_arm.r_forearm_yaw: 14.15,
    reachy.r_arm.r_wrist_pitch: -7.91,
    reachy.r_arm.r_wrist_roll: 45.09,
    reachy.r_arm.r_gripper: -5.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 11.74,
    reachy.r_arm.r_shoulder_roll: -14.54,
    reachy.r_arm.r_arm_yaw: -30.76,
    reachy.r_arm.r_elbow_pitch: -100.55,
    reachy.r_arm.r_forearm_yaw: 28.12,
    reachy.r_arm.r_wrist_pitch: -18.9,
    reachy.r_arm.r_wrist_roll: 49.75,
    reachy.r_arm.r_gripper: -5.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 12.09,
    reachy.r_arm.r_shoulder_roll: -25.35,
    reachy.r_arm.r_arm_yaw: -30.23,
    reachy.r_arm.r_elbow_pitch: -96.5,
    reachy.r_arm.r_forearm_yaw: 34.19,
    reachy.r_arm.r_wrist_pitch: -27.77,
    reachy.r_arm.r_wrist_roll: 55.2,
    reachy.r_arm.r_gripper: -5.54,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 10.77,
    reachy.r_arm.r_shoulder_roll: -34.76,
    reachy.r_arm.r_arm_yaw: -26.63,
    reachy.r_arm.r_elbow_pitch: -93.52,
    reachy.r_arm.r_forearm_yaw: 45.7,
    reachy.r_arm.r_wrist_pitch: -36.21,
    reachy.r_arm.r_wrist_roll: 57.66,
    reachy.r_arm.r_gripper: -60.0,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: 10.24,
    reachy.r_arm.r_shoulder_roll: -48.73,
    reachy.r_arm.r_arm_yaw: -10.72,
    reachy.r_arm.r_elbow_pitch: -91.67,
    reachy.r_arm.r_forearm_yaw: 49.31,
    reachy.r_arm.r_wrist_pitch: -30.94,
    reachy.r_arm.r_wrist_roll: 65.3,
    reachy.r_arm.r_gripper: -60.57,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -5.49,
    reachy.r_arm.r_shoulder_roll: -55.15,
    reachy.r_arm.r_arm_yaw: 13.71,
    reachy.r_arm.r_elbow_pitch: -78.93,
    reachy.r_arm.r_forearm_yaw: 43.86,
    reachy.r_arm.r_wrist_pitch: -26.37,
    reachy.r_arm.r_wrist_roll: 65.39,
    reachy.r_arm.r_gripper: -60.48,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -11.12,
    reachy.r_arm.r_shoulder_roll: -53.74,
    reachy.r_arm.r_arm_yaw: 21.18,
    reachy.r_arm.r_elbow_pitch: -74.27,
    reachy.r_arm.r_forearm_yaw: 43.07,
    reachy.r_arm.r_wrist_pitch: -22.06,
    reachy.r_arm.r_wrist_roll: 64.95,
    reachy.r_arm.r_gripper: -60.55,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -12.08,
    reachy.r_arm.r_shoulder_roll: -52.16,
    reachy.r_arm.r_arm_yaw: 26.54,
    reachy.r_arm.r_elbow_pitch: -71.1,
    reachy.r_arm.r_forearm_yaw: 45.88,
    reachy.r_arm.r_wrist_pitch: -18.19,
    reachy.r_arm.r_wrist_roll: 63.81,
    reachy.r_arm.r_gripper: -60.88,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -14.19,
    reachy.r_arm.r_shoulder_roll: -54.1,
    reachy.r_arm.r_arm_yaw: 25.58,
    reachy.r_arm.r_elbow_pitch: -72.77,
    reachy.r_arm.r_forearm_yaw: 45.53,
    reachy.r_arm.r_wrist_pitch: -16.26,
    reachy.r_arm.r_wrist_roll: 62.14,
    reachy.r_arm.r_gripper: 4.84,
})

#### after block take


poses.append({
    reachy.r_arm.r_shoulder_pitch: -11.64,
    reachy.r_arm.r_shoulder_roll: -51.28,
    reachy.r_arm.r_arm_yaw: 22.15,
    reachy.r_arm.r_elbow_pitch: -71.02,
    reachy.r_arm.r_forearm_yaw: 44.3,
    reachy.r_arm.r_wrist_pitch: -18.63,
    reachy.r_arm.r_wrist_roll: 60.38,
    reachy.r_arm.r_gripper: 20.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -12.43,
    reachy.r_arm.r_shoulder_roll: -51.99,
    reachy.r_arm.r_arm_yaw: 19.78,
    reachy.r_arm.r_elbow_pitch: -72.07,
    reachy.r_arm.r_forearm_yaw: 42.01,
    reachy.r_arm.r_wrist_pitch: -19.78,
    reachy.r_arm.r_wrist_roll: 62.05,
    reachy.r_arm.r_gripper: 20.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -13.84,
    reachy.r_arm.r_shoulder_roll: -51.19,
    reachy.r_arm.r_arm_yaw: 18.19,
    reachy.r_arm.r_elbow_pitch: -75.67,
    reachy.r_arm.r_forearm_yaw: 29.79,
    reachy.r_arm.r_wrist_pitch: -10,
    reachy.r_arm.r_wrist_roll: 63.63,
    reachy.r_arm.r_gripper: 20.49,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -22.63,
    reachy.r_arm.r_shoulder_roll: -43.9,
    reachy.r_arm.r_arm_yaw: 22.06,
    reachy.r_arm.r_elbow_pitch: -78.84,
    reachy.r_arm.r_forearm_yaw: 9.14,
    reachy.r_arm.r_wrist_pitch: 0,
    reachy.r_arm.r_wrist_roll: 65.39,
    reachy.r_arm.r_gripper: 20.45,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -28.87,
    reachy.r_arm.r_shoulder_roll: -34.06,
    reachy.r_arm.r_arm_yaw: 27.42,
    reachy.r_arm.r_elbow_pitch: -78.05,
    reachy.r_arm.r_forearm_yaw: -1.41,
    reachy.r_arm.r_wrist_pitch: 10,
    reachy.r_arm.r_wrist_roll: 65.39,
    reachy.r_arm.r_gripper: 20.4,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -31.16,
    reachy.r_arm.r_shoulder_roll: -24.04,
    reachy.r_arm.r_arm_yaw: 28.3,
    reachy.r_arm.r_elbow_pitch: -76.55,
    reachy.r_arm.r_forearm_yaw: 10.05,
    reachy.r_arm.r_wrist_pitch: 20,
    reachy.r_arm.r_wrist_roll: 65.13,
    reachy.r_arm.r_gripper: 20.45,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -34.93,
    reachy.r_arm.r_shoulder_roll: -10.33,
    reachy.r_arm.r_arm_yaw: 27.86,
    reachy.r_arm.r_elbow_pitch: -72.86,
    reachy.r_arm.r_forearm_yaw: 15.91,
    reachy.r_arm.r_wrist_pitch: 30,
    reachy.r_arm.r_wrist_roll: 62.14,
    reachy.r_arm.r_gripper: 20.45,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -49.0,
    reachy.r_arm.r_shoulder_roll: 5.85,
    reachy.r_arm.r_arm_yaw: 13.89,
    reachy.r_arm.r_elbow_pitch: -60.21,
    reachy.r_arm.r_forearm_yaw: 20.92,
    reachy.r_arm.r_wrist_pitch: 40,
    reachy.r_arm.r_wrist_roll: 55.72,
    reachy.r_arm.r_gripper: 20.45,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -78.09,
    reachy.r_arm.r_shoulder_roll: -3.65,
    reachy.r_arm.r_arm_yaw: -18.28,
    reachy.r_arm.r_elbow_pitch: -45.09,
    reachy.r_arm.r_forearm_yaw:25.27,
    reachy.r_arm.r_wrist_pitch: 40,
    reachy.r_arm.r_wrist_roll: 60.82,
    reachy.r_arm.r_gripper: 20.45,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -86.53,
    reachy.r_arm.r_shoulder_roll: -19.64,
    reachy.r_arm.r_arm_yaw: -19.25,
    reachy.r_arm.r_elbow_pitch: -32.52,
    reachy.r_arm.r_forearm_yaw: 30.6,
    reachy.r_arm.r_wrist_pitch: 40,
    reachy.r_arm.r_wrist_roll: 61.08,
    reachy.r_arm.r_gripper: 20.4,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -91.27,
    reachy.r_arm.r_shoulder_roll: -26.41,
    reachy.r_arm.r_arm_yaw: -19.34,
    reachy.r_arm.r_elbow_pitch: -26.72,
    reachy.r_arm.r_forearm_yaw: 49.35,
    reachy.r_arm.r_wrist_pitch: 40,
    reachy.r_arm.r_wrist_roll: 65.33,
    reachy.r_arm.r_gripper: 20.45,
})
                
poses.append({
    reachy.r_arm.r_shoulder_pitch: -93.91,
    reachy.r_arm.r_shoulder_roll: -26.94,
    reachy.r_arm.r_arm_yaw: -21.62,
    reachy.r_arm.r_elbow_pitch: -22.94,
    reachy.r_arm.r_forearm_yaw: 49.96,
    reachy.r_arm.r_wrist_pitch: 40,
    reachy.r_arm.r_wrist_roll: 59.33,
    reachy.r_arm.r_gripper: -17.45,
})


def happy_antennas():
    reachy.head.l_antenna.speed_limit = 0.0
    reachy.head.r_antenna.speed_limit = 0.0
    
    for _ in range(9):
        reachy.head.l_antenna.goal_position = 10.0
        reachy.head.r_antenna.goal_position = -10.0

        time.sleep(0.1)

        reachy.head.l_antenna.goal_position = -10.0
        reachy.head.r_antenna.goal_position = 10.0

        time.sleep(0.1)
    
    reachy.head.l_antenna.goal_position = 0.0
    reachy.head.r_antenna.goal_position = 0.0
    
def down_antennas():
    reachy.head.l_antenna.speed_limit = 40.0
    reachy.head.r_antenna.speed_limit = 40.0
    reachy.head.l_antenna.goal_position = 20.0
    reachy.head.r_antenna.goal_position = -20.0

def up_antennas():
    reachy.head.l_antenna.speed_limit = 40.0
    reachy.head.r_antenna.speed_limit = 40.0
    reachy.head.l_antenna.goal_position = 0.0
    reachy.head.r_antenna.goal_position = 0.0


def open_gripper():
    goto(goal_positions= {reachy.r_arm.r_gripper: -60.55,},duration=0.4,interpolation_mode=InterpolationMode.LINEAR)

def close_gripper():
    goto(goal_positions= {reachy.r_arm.r_gripper: 14.0,},duration=0.4,interpolation_mode=InterpolationMode.LINEAR)

def turn_on_fans():
    reachy.fans.l_shoulder_fan.on()
    reachy.fans.l_elbow_fan.on()
    reachy.fans.l_wrist_fan.on()
    reachy.fans.r_shoulder_fan.on()
    reachy.fans.r_elbow_fan.on()
    reachy.fans.r_wrist_fan.on()
    reachy.fans.l_antenna_fan.on() 
    reachy.fans.r_antenna_fan.on() 
def turn_off_fans():
    reachy.fans.l_shoulder_fan.off()
    reachy.fans.l_elbow_fan.off()
    reachy.fans.l_wrist_fan.off()
    reachy.fans.r_shoulder_fan.off()
    reachy.fans.r_elbow_fan.off()
    reachy.fans.r_wrist_fan.off()
    reachy.fans.l_antenna_fan.off() 
    reachy.fans.r_antenna_fan.off() 

def look_at_human():
    # reachy.head.look_at(
    # x=-0.5,
    # y=2,
    # z=0,
    # duration=1.0,
    # starting_positions={
    #     reachy.head.neck_roll: reachy.head.neck_roll.goal_position,
    #     reachy.head.neck_pitch: reachy.head.neck_pitch.goal_position,
    #     reachy.head.neck_yaw: reachy.head.neck_yaw.goal_position
    # })

    pose = {reachy.head.neck_pitch: 2,
    reachy.head.neck_roll: -10.53,
    reachy.head.neck_yaw: 8.79,}
    goto(goal_positions=pose,duration=1,interpolation_mode=InterpolationMode.LINEAR)


happy_antennas()

# reachy.head.look_at(x=x, y=y, z=z, duration=1.0)

goto(goal_positions= poses[0],duration=1,interpolation_mode=InterpolationMode.MINIMUM_JERK)

for idx,pose in enumerate(poses):
    
    # reachy.head.look_at(x=x, y=y, z=z, duration=0.1)
    # if idx == len(poses)-1:
    #     input()
    
    goto(goal_positions= pose,duration=0.4,interpolation_mode=InterpolationMode.LINEAR)
input("Waiting for press Enter")

sleep(5)
goto(goal_positions= poses[0],duration=5,interpolation_mode=InterpolationMode.MINIMUM_JERK)
goto(goal_positions= poses[1],duration=1,interpolation_mode=InterpolationMode.MINIMUM_JERK)
# sleep(5)

reachy.turn_off_smoothly('reachy')


#alles in try
"""
import traceback

if __name__ == "__main__":
            
    print("Started")
    turn_on_fans()

    try:
            CODE STARTS HERE
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        reachy.turn_off_smoothly('reachy')
    except:
        print("Error")
        reachy.turn_off_smoothly('reachy')
        print(traceback.format_exc())

    reachy.turn_off_smoothly('reachy')
    
    """
