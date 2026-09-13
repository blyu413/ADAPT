"""Unitree HG LowCmd CRC, using the SDK's pure-Python polynomial algorithm.

Derived from unitree_sdk2py/utils/crc.py.
Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
SPDX-License-Identifier: BSD-3-Clause
See UNITREE_SDK_LICENSE in this directory.

Only the 35-motor HG command layout is supported. This intentionally has no
native-library loader: the upstream wheel omits its CRC shared libraries.
"""

import struct


def command_crc(cmd):
    values = [cmd.mode_pr, cmd.mode_machine]
    for motor in cmd.motor_cmd:
        values.extend([motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd, motor.reserve])
    values.extend(cmd.reserve)
    # Exclude the trailing crc word, exactly as the SDK's 1004-byte packing does.
    packed = struct.pack("<2B2x" + "B3x5fI" * 35 + "4I", *values)
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for (word,) in struct.iter_unpack("<I", packed):
        bit = 1 << 31
        for _ in range(32):
            top = crc & 0x80000000
            crc = (crc << 1) & 0xFFFFFFFF
            if top:
                crc ^= polynomial
            if word & bit:
                crc ^= polynomial
            bit >>= 1
    return crc
