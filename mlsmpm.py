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
    # 質点の初期化。XYZ方向に伸ばしている
    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i * 0.1 for j in ti.static(range(3))]

    window = ti.ui.Window("Taichi Particle view", (1024, 1024), vsync=True)

    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # シーンの設定。カメラの位置、ライトの設定、メッシュの描画
    camera.position(5, 2, 2)
    camera.lookat(0.0, 0.0, 0)

    # パーティクル作成
    N = 50
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    init_points_pos(particles_pos)

    while window.running:
        # キーボードによる移動に対応
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.02)
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
