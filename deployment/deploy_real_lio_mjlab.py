"""G1 state inspection and explicitly gated low-level control. Hardware test required."""

import argparse
import time

import mujoco
import numpy as np

from common.telemetry import Telemetry, observer_record
from observer import MomentumObserver
from common.paths import ASSET_DIR, checkpoint_path
from deployment.config_manager import configure_model
from deployment.robot_interface import G1
from deployment.policy_manager import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", required=True, help="Robot Ethernet interface")
    parser.add_argument(
        "--lidar-frame",
        required=True,
        choices=["raw", "compensated"],
        help="Whether the Livox driver already compensates the upside-down mount",
    )
    parser.add_argument("--lio-topic", default="rt/Odometry_imu_local")
    parser.add_argument("--policy", choices=["adapt", "baseline", "lightstep"], default="adapt")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--state-only", action="store_true", help="Read lowstate + LIO without requiring lowcmd"
    )
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="Create a motor publisher after gamepad confirmation",
    )
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--log")
    parser.add_argument("--udp")
    args = parser.parse_args()
    if args.state_only and args.enable_control:
        parser.error("--state-only and --enable-control are mutually exclusive")
    if args.state_only and (args.log or args.udp):
        parser.error("--state-only prints state; observer logging requires observed lowcmd")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    policy = Policy(args.checkpoint or checkpoint_path(args.policy))
    cfg = policy.cfg
    model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "g1_29dof.xml"))
    actuators = configure_model(model, cfg)
    observer = MomentumObserver(
        model,
        cfg.control_dt,
        cfg.observer_gain,
        filter_order=cfg.filter_order,
        cutoff_hz=cfg.cutoff_hz,
    )
    robot = G1(cfg, model, args.net, args.lidar_frame, args.lio_topic)
    telemetry = Telemetry(args.log, args.udp)
    gamepad = None
    try:
        print("Read-only: waiting up to 10s for lowstate and advancing LIO timestamps...")
        deadline = time.monotonic() + 10
        while True:
            try:
                state, status = robot.snapshot()
                break
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        targets = None
        if args.enable_control:
            from common.joystick import Gamepad

            gamepad = Gamepad()
            print(
                "Suspend/support G1; release other controllers manually. Hold LB + press START to stand."
            )
            while True:
                buttons, _ = gamepad.read()
                robot.snapshot()
                if buttons["stop"]:
                    return
                if buttons["start"] and buttons["deadman"]:
                    break
                time.sleep(cfg.control_dt)
            robot.enable_control()
            initial = robot.snapshot()[0].qpos[7:]
            default = np.asarray(cfg.default_position)
            print(
                "Keep LB held. Moving to default posture over 2s; press A afterwards to run policy."
            )
            ramp_start = time.monotonic()
            while True:
                start = time.monotonic()
                buttons, _ = gamepad.read()
                if buttons["stop"] or not buttons["deadman"]:
                    return
                alpha = min(1, (start - ramp_start) / 2)
                targets = (1 - alpha) * initial + alpha * default
                robot.send(targets)
                if alpha == 1 and buttons["arm"]:
                    break
                time.sleep(max(0, cfg.control_dt - (time.monotonic() - start)))
        state, _ = robot.snapshot()
        observer.reset(state.qpos, state.qvel)
        policy.history.reset()
        begin = time.monotonic()
        step = 0
        while time.monotonic() - begin < args.seconds:
            start = time.monotonic()
            state, status = robot.snapshot()
            command = np.zeros(3, np.float32)
            if gamepad:
                buttons, command = gamepad.read()
                if buttons["stop"] or not buttons["deadman"]:
                    break
            if args.state_only:
                if step % 50 == 0:
                    print(
                        f"mode={status['mode_machine']} pelvis velocity={state.linear_velocity.round(3)} {status}"
                    )
            else:
                if targets is None or not args.enable_control:
                    targets = robot.observed_targets()
                    policy.history.last_action[:] = (
                        targets - np.asarray(cfg.default_position)
                    ) / np.asarray(cfg.action_scale)
                ctrl = np.zeros(29)
                ctrl[actuators] = targets
                observer.update(state.qpos, state.qvel, ctrl)
                local = observer.local_residual()
                proposed = policy.step(state, command, local)
                if args.enable_control:
                    robot.send(proposed)
                    targets = proposed
                telemetry.write(
                    observer_record(
                        observer,
                        policy,
                        state,
                        command,
                        local,
                        source="g1",
                        time=start - begin,
                        **status,
                    )
                )
            step += 1
            time.sleep(max(0, cfg.control_dt - (time.monotonic() - start)))
    finally:
        try:
            robot.close()
        finally:
            telemetry.close()
            if gamepad:
                gamepad.close()


if __name__ == "__main__":
    main()
