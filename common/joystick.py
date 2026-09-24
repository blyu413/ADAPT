"""The source deployment's wired Xbox layout, with an LB hold-to-run gate."""

import numpy as np


class Gamepad:
    def __init__(self):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() != 1:
            raise RuntimeError("Connect exactly one wired Xbox-compatible controller")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

    def read(self):
        self.pygame.event.pump()
        if not self.joy.get_attached():
            raise RuntimeError("Gamepad disconnected")
        buttons = dict(
            start=bool(self.joy.get_button(7)),
            select=bool(self.joy.get_button(6)),
            arm=bool(self.joy.get_button(0)),
            stop=bool(self.joy.get_button(1)),
            deadman=bool(self.joy.get_button(4)),
        )
        command = -np.array(
            [self.joy.get_axis(1), self.joy.get_axis(0), self.joy.get_axis(2)], np.float32
        )
        command[np.abs(command) < 0.1] = 0
        return buttons, command

    def close(self):
        self.pygame.quit()
