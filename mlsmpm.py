import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple

import taichi as ti

from opt import TaichiArch

logger = logging.getLogger(__name__)


@dataclass
class StaticBall:
    radius: float
    center: ti.Vector
    color: Tuple[float]

    def __init__(
        self, radius: float, center: List[float], color: Tuple[float] = (0.5, 0.42, 0.8)
    ):
        self.radius = radius
        self.center = ti.Vector.field(3, dtype=float, shape=(1,))
        self.center[0] = center
        self.color = color

    def draw(self, scene: ti.ui.Scene):
        scene.particles(self.center, radius=self.radius, color=self.color)


def main():
    window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)

    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # ボールの初期設定
    ball = StaticBall(0.3, [0.0, 0.0, 0.0])

    while window.running:
        # シーンの設定。カメラの位置、ライトの設定、メッシュの描画
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        ball.draw(scene)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    parser = argparse.ArgumentParser(prog="MLS-MPM Incremental")

    parser.add_argument(
        "--arch",
        type=TaichiArch,
        choices=list(TaichiArch),
        help="Taichi Arch",
        default=TaichiArch.cpu,
    )

    opts = parser.parse_args()

    logger.info(f"Taichi Arch: {opts.arch}")
    ti.init(arch=opts.arch.decode())
    main()
