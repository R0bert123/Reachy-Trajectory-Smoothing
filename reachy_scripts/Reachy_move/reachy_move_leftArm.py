from reachy_sdk import ReachySDK
import time

def main():
    reachy = ReachySDK(host="localhost")
    print("Linker Arm wird bewegt")

    reachy.l_arm.shoulder_pitch.goto(
        0.5,
        duration=1,
    )

    time.sleep(1)

    print("Gehe züruck auf neutrale Position")
    reachy.l_arm.shoulder_pitch.goto(
        0.0,
        duration=1,
    )

if __name__ == "__main__":
    main()