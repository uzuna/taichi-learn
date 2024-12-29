import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)


# モジュール分割
# これらはコンパイル時定数として扱うことになるので変数化を行う

# 点数
n = 128
# 配置間隔
quad_size = 1.0 / n
# シミュレーションの時間刻み。点数に応じて変化させるのはなぜ?
# 1/60にかかっているので、128以上の大きさでなければsubstepsが不適切な値になりそう?
dt = 4e-2 / n
# Taichi並列数? 初期設定では53になる
substeps = int(1 / 60 // dt)

# 重力
GRAVITY = [0, -9.8, 0]
# ばね定数。小さくすることで柔らかく反発が強くなる
spring_Y = 3e4
# 質点に影響する他の質点の影響で減速する係数。
dashpot_damping = 1e4
# バネの抵抗による速度減衰。ボールとの接触時のみ発生させているので接触抵抗
drag_damping = 1

# ボールの半径
ball_radius = 0.3
# ボールの中心
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

# 位置と速度の保持
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

# ポリゴン向けの定数群
# ポリゴンの数は1区画あたり2つ使う。面は最低2点が必要なので面数(n-1^2)に分割数2をかける
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# ばねを曲げるかどうか?
bending_springs = False


# 質点の初期化
@ti.kernel
def initialize_mass_points():
    # 結果が変化するように位置をランダムでずらす
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


# メッシュのインデックスを初期化。レンダリングのための情報なので、初期化のみ
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

initialize_mesh_indices()

# 影響するバネのインデックスを作成
spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

# シミュレーションのサブステップ
@ti.kernel
def substep(g_x: float, g_y: float, g_z: float, spring_y: float, dp_damping: float, drag_damping:float):
    g = ti.Vector([g_x, g_y, g_z])
    # 重力を適用
    for i in ti.grouped(x):
        v[i] += g * dt


    for i in ti.grouped(x):
        # 力の大きさを保持
        force = ti.Vector([0.0, 0.0, 0.0])
        # 周囲の質点とのバネによる力の変化を計算
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dp_damping * quad_size
        # 力を速度に変換
        v[i] += force * dt

    # 移動抵抗とボールとの衝突処理
    for i in ti.grouped(x):
        # ボールとの衝突処理。座標内に入ったら中心に向かうベクトル成分だけを0にする
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            # 接触抵抗を考えるなら、ここで速度に関する減速の大きさをベクトルとの法線成分から計算するのがいいかも?
            v[i] -= min(v[i].dot(normal), 0) * normal
            # 移動抵抗分減速
            v[i] *= ti.exp(-drag_damping * dt)
        # 位置を更新
        x[i] += dt * v[i]

# レンダリング用の頂点情報を更新
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
gui = window.get_gui()

canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

with gui.sub_window("Sub Window", x=10, y=10, width=300, height=100):
    gui.text("text")
    is_clicked = gui.button("name")


# 実行時変数定義
current_t = 0.0
max_t = 1.5
paused=False

# 質点の初期化
initialize_mass_points()

# サブステップ数の表示
print("Substeps: ", substeps)


def init():
    global current_t
    
    initialize_mass_points()
    initialize_mesh_indices()

# オプション表示
def show_options():
    global max_t
    global spring_Y
    global dashpot_damping
    global drag_damping
    global paused

    with gui.sub_window("Gravity", 0.05, 0.2, 0.2, 0.1) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Cloth", 0.05, 0.3, 0.2, 0.1) as w:
        spring_Y = w.slider_float("spring", spring_Y, 1e2, 26e3)
        dashpot_damping = w.slider_float("dashpot", dashpot_damping, 1e2, 32e3)
        drag_damping = w.slider_float("drag", drag_damping, 0.8, 100)

    with gui.sub_window("Options", 0.05, 0.45, 0.2, 0.4) as w:
        max_t = w.slider_float("max t", max_t, 0.5, 5.0)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True
        if w.button("Reset"):
            max_t = 1.5
            spring_Y = 3e4
            dashpot_damping = 1e4
            drag_damping = 1


# 実行ループ
while window.running:
    # 時間経過で位置をリセット
    if current_t > max_t:
        # Reset
        initialize_mass_points()
        current_t = 0

    # substep分割数分計算を進める
    if not paused:
        for i in range(substeps):
            substep(*GRAVITY, spring_Y, dashpot_damping, drag_damping)
            current_t += dt
    # レンダリング用の頂点情報を更新
    update_vertices()

    # シーンの設定。カメラの位置、ライトの設定、メッシュの描画
    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    # 衝突処理用のボールを描画
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    show_options()
    window.show()
