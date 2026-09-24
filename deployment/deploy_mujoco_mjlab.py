"""Minimal flat-ground G1 deployment, independent of mjlab and the robot SDK."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import numpy as np

from common.telemetry import Telemetry, observer_record
from common.paths import ASSET_DIR, checkpoint_path
from deployment.mujoco_robot import MujocoRobot
from deployment.policy_manager import PolicyManager, load_policy


class Simulation:
    def __init__(
        self,
        checkpoint,
        *,
        policy_name="policy",
        additional_policies=(),
        seed=0,
        encoder_noise=False,
    ):
        self.policy_manager = PolicyManager()
        self.register_policy(policy_name, checkpoint, set_active=True)
        for name, path in additional_policies:
            self.register_policy(name, path)
        self.robot = MujocoRobot(self.cfg, seed=seed, encoder_noise=encoder_noise)
        self.reset()

    @property
    def policy(self):
        return self.policy_manager.active_policy

    @property
    def cfg(self):
        return self.policy.cfg

    @property
    def observer(self):
        return self.policy.observer

    def register_policy(self, name, checkpoint, *, set_active=False):
        policy = load_policy(checkpoint, name=name)
        nominal = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene.xml"))
        policy.attach_observer(nominal)
        self.policy_manager.register(policy, set_active=set_active and not hasattr(self, "robot"))
        if set_active and hasattr(self, "robot"):
            self.set_active(name)

    def set_active(self, name):
        """Switch without resetting the robot pose or retaining old policy history."""
        self.policy_manager.set_active(name, self.robot)
        return name

    def next_policy(self):
        return self.policy_manager.next(self.robot)

    def prev_policy(self):
        return self.policy_manager.prev(self.robot)

    def reset(self):
        state = self.robot.reset()
        self.policy.reset(state)

    def local_residual(self):
        rotation = self.robot.data.xmat[self.observer.body_id].reshape(3, 3)
        return self.observer.local_residual(rotation)

    def step(self, command):
        """One tick: observation/policy -> robot control -> encoder/MoMo refresh."""
        state = self.robot.get_state()
        targets = self.policy_manager.step(state, command)
        state = self.robot.step(targets)
        self.policy_manager.refresh_encoder(state)
        self.policy.update_momo(state, targets)
        extra = (
            {"policy_name": self.policy_manager.active_policy_name}
            if len(self.policy_manager.list_policies()) > 1
            else {}
        )
        return observer_record(
            self.observer,
            self.policy,
            state,
            command,
            self.local_residual(),
            source="mujoco",
            time=float(self.robot.data.time),
            root_height=float(self.robot.data.qpos[2]),
            **extra,
        )


def additional_policy(value):
    """A released policy name or NAME=ONNX_PATH for a custom export."""
    if "=" in value:
        name, path = value.split("=", 1)
        if not name or not path:
            raise argparse.ArgumentTypeError("Use NAME=ONNX_PATH")
        return name, Path(path)
    try:
        return value, checkpoint_path(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["adapt", "baseline", "lightstep"], default="adapt")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--add-policy",
        action="append",
        type=additional_policy,
        default=[],
        metavar="NAME[=ONNX_PATH]",
        help="Register another policy for runtime switching",
    )
    parser.add_argument(
        "--command", nargs=3, type=float, default=[0.8, 0, 0], metavar=("VX", "VY", "WZ")
    )
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--controller", action="store_true", help="Use an Xbox controller for commands and switching"
    )
    parser.add_argument("--encoder-noise", action="store_true", help="Shared Stage 1 encoder noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--udp", help="Telemetry host:port, e.g. 127.0.0.1:9870")
    args = parser.parse_args()
    policy_name = Path(args.checkpoint).stem if args.checkpoint else args.policy
    sim = Simulation(
        args.checkpoint or checkpoint_path(args.policy),
        policy_name=policy_name,
        additional_policies=args.add_policy,
        seed=args.seed,
        encoder_noise=args.encoder_noise,
    )
    steps = int(args.seconds / sim.cfg.control_dt)
    if steps < 1:
        raise ValueError("--seconds must cover at least one policy step")
    controller = None
    if args.controller:
        from common.joystick import Gamepad

        try:
            controller = Gamepad()
        except ImportError:
            parser.error("Install the controller extra: uv sync --frozen --extra controller")
        except RuntimeError as exc:
            parser.error(str(exc))
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer

        context = mujoco.viewer.launch_passive(sim.robot.model, sim.robot.data)
    telemetry = Telemetry(args.log, args.udp)
    if controller and len(sim.policy_manager.list_policies()) > 1:
        print("Policies:", ", ".join(sim.policy_manager.list_policies()), "(START/SELECT to switch)")
    min_height = float("inf")
    step_count = 0
    previous_start = previous_select = False
    try:
        with context as viewer:
            while True:
                if not args.add_policy and step_count >= steps:
                    break
                start = time.monotonic()
                if viewer and not viewer.is_running():
                    break
                command = np.asarray(args.command, np.float32)
                if controller:
                    buttons, command = controller.read()
                    if len(sim.policy_manager.list_policies()) > 1:
                        if previous_start and not buttons["start"]:
                            print("Active policy:", sim.next_policy())
                        if previous_select and not buttons["select"]:
                            print("Active policy:", sim.prev_policy())
                    previous_start = buttons["start"]
                    previous_select = buttons["select"]
                if (
                    args.add_policy
                    and sim.robot.data.time + sim.cfg.control_dt > args.seconds + 1e-9
                ):
                    break
                record = sim.step(command)
                step_count += 1
                telemetry.write(record)
                min_height = min(min_height, record["root_height"])
                if not np.isfinite(sim.robot.data.qpos).all():
                    raise RuntimeError("Simulation state became non-finite")
                if viewer:
                    viewer.sync()
                if viewer or controller:
                    time.sleep(max(0, sim.cfg.control_dt - (time.monotonic() - start)))
    finally:
        telemetry.close()
        if controller:
            controller.close()
    print(
        f"{sim.cfg.task}: simulated {sim.robot.data.time:.2f}s, "
        f"minimum root height {min_height:.3f}m"
    )


if __name__ == "__main__":
    main()
