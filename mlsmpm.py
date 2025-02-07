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


@ti.data_oriented
class Particle:
    """
    パーティクルのクラス
    データを保持し、ti.kernelのメソッドを使うためにはti.data_orientedデコレーとが必要
    """

    pos: ti.Vector
    vel: ti.Vector
    mass: float
    n: int

    def __init__(self, n: int):
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=pow(n, 2))
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=pow(n, 2))
        self.mass = ti.field(dtype=ti.f32, shape=pow(n, 2))
        self.n = n

    @ti.kernel
    def init_points_pos(self, width: ti.f32, n: ti.i32):
        """
        配列の初期化
        @param points: 配列
        @param width: グリッド配置の幅の長さ
        @param n: グリッドの分割数
        """
        quad_size = width / n
        half_width = width / 2
        # 結果が変化するように位置を10%のランダムを与える
        random_offset = (
            ti.Vector(
                [ti.random() * width - half_width, ti.random() * width - half_width]
            )
            * 0.1
        )

        for i in range(n):
            for j in range(n):
                self.pos[i * n + j] = [
                    i * quad_size - half_width + random_offset[0],
                    0.6,
                    j * quad_size - half_width + random_offset[1],
                ]
                # 速度は適当な値を入れる
                self.vel[i * n + j] = [0.01, 0.02, -0.05]


def main():
    @ti.kernel
    def substep(pos: ti.template(), vel: ti.template(), dt: float):
        """
        シミュレーションのステップ
        @param p: パーティクル
        @param dt: シミュレーションの時間刻み
        """

        # 速度を反映
        for i in ti.grouped(pos):
            pos[i] += vel[i] * dt

    window = ti.ui.Window("Taichi Particle view", (1024, 1024), vsync=True)

    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # シーンの設定。カメラの位置、ライトの設定、メッシュの描画
    camera.position(5, 2, 2)
    camera.lookat(0.0, 0.0, 0)

    # パーティクル作成
    N = 16
    p = Particle(N)
    p.init_points_pos(1.0, N)
    # Cell生成
    # c_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
    # c_mass = ti.field(dtype=ti.f32, shape=())

    current_t = 0.0
    dt = 0.01

    while window.running:
        # キーボードによる移動に対応
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        substep(p.pos, p.vel, dt)
        current_t += dt

        scene.particles(p.pos, color=(0.68, 0.26, 0.19), radius=0.02)
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
