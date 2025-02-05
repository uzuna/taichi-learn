from enum import Enum

import taichi as ti


class TaichiArch(Enum):
    cpu = "cpu"
    opengl = "opengl"
    vulkan = "vulkan"

    def __str__(self):
        return self.value

    def decode(self):
        if self == TaichiArch.cpu:
            return ti.cpu
        elif self == TaichiArch.opengl:
            return ti.opengl
        elif self == TaichiArch.vulkan:
            return ti.vulkan
