"""G1 HG messages only; SDK's serialized velocity field is named dq."""


def initialize_command(cmd, mode_machine):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = 0  # PR joint control, not parallel AB motors.
    for motor in cmd.motor_cmd:
        motor.mode = 0
        motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0


def position_command(cmd, mapping, targets, stiffness, damping):
    for i, index in enumerate(mapping):
        motor = cmd.motor_cmd[index]
        motor.mode = 1
        motor.q = float(targets[i])
        motor.dq = motor.tau = 0.0
        motor.kp = float(stiffness[i])
        motor.kd = float(damping[i])


def damping_command(cmd, mapping):
    for index in mapping:
        motor = cmd.motor_cmd[index]
        motor.mode = 1
        motor.q = motor.dq = motor.tau = motor.kp = 0.0
        motor.kd = 8.0
