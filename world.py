"""
world.py —— "真实世界"

一个严格遵守牛顿力学的质点宇宙（默认二维，代码本身与维数无关）：

  * 任意两个质点之间有万有引力        F = G · m_i · m_j / r²
  * 部分质点对之间连着弹簧（胡克定律）  F = k · (r − L)
  * 每个质点的加速度                  a_i = (它受到的合力) / m_i

神经网络看不到这个文件里的任何公式。它能看到的只有"观测"：
每个物体的质量读数 m、每根弹簧的两个读数 (k, L)，以及各个时刻物体的位置。
"""
import numpy as np

G = 1.0  # 引力常数。模型不知道这个数，要自己从数据里学出来。


# --------------------------------------------------------------------------
# 真实的物理定律
# --------------------------------------------------------------------------
def pair_force_true(mi, mj, r, k, L):
    """真实的成对吸引力大小（正 = 相吸，负 = 相斥）：万有引力 + 弹簧。"""
    return G * mi * mj / r**2 + k * (r - L)


def accelerations(x, m, K, L):
    """真实加速度。
    x: (B,N,D) 位置   m: (B,N) 质量   K, L: (B,N,N) 弹簧劲度/原长（K=0 表示没有弹簧）
    """
    N = x.shape[1]
    d = x[:, None, :, :] - x[:, :, None, :]            # d[b,i,j] = x_j − x_i
    r = np.linalg.norm(d, axis=-1)
    eye = np.eye(N, dtype=bool)
    r = np.where(eye, 1.0, r)
    f = pair_force_true(m[:, :, None], m[:, None, :], r, K, L)
    f = np.where(eye, 0.0, f)
    F = (f[..., None] * d / r[..., None]).sum(axis=2)  # 质点 i 受到的合力
    return F / m[..., None]


def energy(x, v, m, K, L):
    """总机械能 = 动能 + 引力势能 + 弹性势能。返回 (B,)"""
    N = x.shape[1]
    iu = np.triu_indices(N, 1)
    d = x[:, None, :, :] - x[:, :, None, :]
    r = np.linalg.norm(d, axis=-1)[:, iu[0], iu[1]]
    mm = (m[:, :, None] * m[:, None, :])[:, iu[0], iu[1]]
    kin = 0.5 * (m * (v**2).sum(-1)).sum(-1)
    pot = (-G * mm / r + 0.5 * K[:, iu[0], iu[1]] * (r - L[:, iu[0], iu[1]]) ** 2).sum(-1)
    return kin + pot


def momentum(v, m):
    """总动量 (B,D)"""
    return (m[..., None] * v).sum(axis=1)


def angular_momentum(x, v, m):
    """总角动量（仅二维）(B,)"""
    return (m * (x[..., 0] * v[..., 1] - x[..., 1] * v[..., 0])).sum(axis=1)


def pair_distances(x):
    """所有质点对的距离 (B, N(N-1)/2)"""
    N = x.shape[1]
    iu = np.triu_indices(N, 1)
    d = x[:, None, :, :] - x[:, :, None, :]
    return np.linalg.norm(d, axis=-1)[:, iu[0], iu[1]]


# --------------------------------------------------------------------------
# 数值积分（四阶 Runge–Kutta）。acc_fn 可以换成任何"给位置、出加速度"的函数，
# 所以评估时神经网络的推演和真实世界用的是同一个积分器。
# --------------------------------------------------------------------------
def rk4_step(x, v, dt, acc_fn):
    a1 = acc_fn(x)
    x2 = x + 0.5 * dt * v
    v2 = v + 0.5 * dt * a1
    a2 = acc_fn(x2)
    x3 = x + 0.5 * dt * v2
    v3 = v + 0.5 * dt * a2
    a3 = acc_fn(x3)
    x4 = x + dt * v3
    v4 = v + dt * a3
    a4 = acc_fn(x4)
    x_new = x + dt / 6.0 * (v + 2 * v2 + 2 * v3 + v4)
    v_new = v + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    return x_new, v_new


def simulate(x, v, acc_fn, T, dt=1e-3, record_every=10, r_dead=0.1):
    """从 (x, v) 出发推演时长 T。每 record_every 步记录一帧。

    返回 xs, vs: (B, 帧数, N, D)；alive: (B, 帧数)。
    alive 为 False 表示该系统在此之前已有两个质点靠得比 r_dead 还近——
    点质量引力在 r→0 处发散，数值上不可信，此后的帧一律丢弃（系统被冻结）。
    """
    x, v = x.copy(), v.copy()
    alive = np.ones(x.shape[0], dtype=bool)
    xs, vs, al = [x.copy()], [v.copy()], [alive.copy()]
    steps = int(round(T / dt))
    for s in range(steps):
        x_new, v_new = rk4_step(x, v, dt, acc_fn)
        alive &= np.isfinite(x_new).all(axis=(1, 2)) & np.isfinite(v_new).all(axis=(1, 2))
        x = np.where(alive[:, None, None], x_new, x)
        v = np.where(alive[:, None, None], v_new, v)
        alive &= pair_distances(x).min(axis=1) >= r_dead
        if (s + 1) % record_every == 0:
            xs.append(x.copy()); vs.append(v.copy()); al.append(alive.copy())
    return np.stack(xs, 1), np.stack(vs, 1), np.stack(al, 1)


# --------------------------------------------------------------------------
# 随机场景
# --------------------------------------------------------------------------
SCENE_DEFAULTS = dict(
    box=2.5,            # 初始位置均匀分布在 [-box, box]^D
    min_sep=0.8,        # 初始最小间距
    m_range=(0.25, 4.0),  # 质量范围（对数均匀）
    v_std=0.5,          # 初速度各分量的标准差
    p_link=(0.0, 1.0),  # 两个质点之间连弹簧的概率；给区间则每个场景各抽一个（有的系统全是弹簧，有的一根都没有）
    link_max_r=3.5,     # 只有初始距离小于它的质点对才可能连弹簧
    k_range=(0.5, 4.0),   # 弹簧劲度范围
    L_range=(0.5, 2.5),   # 弹簧原长范围
)


def sample_scene(rng, B, N, D=2, **kw):
    """随机生成 B 个 N 体系统的初始状态和属性。"""
    p = {**SCENE_DEFAULTS, **kw}
    x = rng.uniform(-p["box"], p["box"], size=(B, N, D))
    if N > 1:
        for _ in range(10000):                          # 拒绝采样：保证初始间距
            bad = pair_distances(x).min(axis=1) < p["min_sep"]
            if not bad.any():
                break
            x[bad] = rng.uniform(-p["box"], p["box"], size=(bad.sum(), N, D))
    m = np.exp(rng.uniform(np.log(p["m_range"][0]), np.log(p["m_range"][1]), size=(B, N)))
    v = rng.normal(0.0, p["v_std"], size=(B, N, D))
    v -= (m[..., None] * v).sum(1, keepdims=True) / m.sum(1)[:, None, None]   # 质心系
    x -= (m[..., None] * x).sum(1, keepdims=True) / m.sum(1)[:, None, None]

    K = np.zeros((B, N, N)); L = np.zeros((B, N, N))
    d = x[:, None, :, :] - x[:, :, None, :]
    r0 = np.linalg.norm(d, axis=-1)
    iu = np.triu_indices(N, 1)
    p_link = rng.uniform(*p["p_link"], size=(B, 1)) if isinstance(p["p_link"], tuple) else p["p_link"]
    link = (rng.random((B, len(iu[0]))) < p_link) & (r0[:, iu[0], iu[1]] < p["link_max_r"])
    k = rng.uniform(*p["k_range"], size=link.shape) * link
    l = rng.uniform(*p["L_range"], size=link.shape) * link
    K[:, iu[0], iu[1]] = k; K[:, iu[1], iu[0]] = k
    L[:, iu[0], iu[1]] = l; L[:, iu[1], iu[0]] = l
    return x, v, m, K, L


# --------------------------------------------------------------------------
# 观测数据：只有位置序列。加速度由相邻三帧位置做二阶差分估计出来。
# --------------------------------------------------------------------------
def observe(rng, B, N, T=3.0, dt_obs=0.01, frame_stride=8, r_min=0.4, r_max=8.0,
            sim_substeps=5, **scene_kw):
    """模拟 B 个 N 体系统并"观测"它们。

    返回一个字典（S 个样本）：
      x (S,N,D)  m (S,N)  K,L (S,N,N)   —— 模型的输入
      a_obs (S,N,D)  —— 由位置序列做二阶中心差分得到的加速度（训练时唯一的监督信号）
      a_true (S,N,D) —— 真实加速度（只用于评估，训练时不看）
    只保留所有质点对距离都在 [r_min, r_max] 内的帧。
    """
    x0, v0, m, K, L = sample_scene(rng, B, N, **scene_kw)
    acc = lambda x: accelerations(x, m, K, L)
    xs, vs, alive = simulate(x0, v0, acc, T, dt=dt_obs / sim_substeps, record_every=sim_substeps)
    F = xs.shape[1]
    a_fd = (xs[:, 2:] - 2 * xs[:, 1:-1] + xs[:, :-2]) / dt_obs**2          # 帧 1..F-2
    frames = np.arange(1, F - 1, frame_stride)
    out = {k: [] for k in ("x", "m", "K", "L", "a_obs", "a_true", "traj", "frame")}
    for t in frames:
        xt = xs[:, t]
        pd = pair_distances(xt)
        ok = alive[:, t + 1] & (pd.min(1) >= r_min) & (pd.max(1) <= r_max)
        if not ok.any():
            continue
        out["x"].append(xt[ok]); out["m"].append(m[ok]); out["K"].append(K[ok]); out["L"].append(L[ok])
        out["a_obs"].append(a_fd[ok, t - 1])
        out["a_true"].append(accelerations(xt[ok], m[ok], K[ok], L[ok]))
        out["traj"].append(np.nonzero(ok)[0]); out["frame"].append(np.full(ok.sum(), t))
    return {k: np.concatenate(val) for k, val in out.items()}


def pad(data, n_max):
    """把 N 体数据补零到 n_max 个槽位，并附上 mask（哪些槽位里真的有物体）。"""
    S, N = data["m"].shape
    D = data["x"].shape[-1]
    out = {}
    for key in ("x", "a_obs", "a_true"):
        arr = np.zeros((S, n_max, D), dtype=np.float32); arr[:, :N] = data[key]; out[key] = arr
    arr = np.zeros((S, n_max), dtype=np.float32); arr[:, :N] = data["m"]; out["m"] = arr
    for key in ("K", "L"):
        arr = np.zeros((S, n_max, n_max), dtype=np.float32); arr[:, :N, :N] = data[key]; out[key] = arr
    mask = np.zeros((S, n_max), dtype=bool); mask[:, :N] = True; out["mask"] = mask
    return out


def concat(datasets):
    return {k: np.concatenate([d[k] for d in datasets]) for k in datasets[0]}


if __name__ == "__main__":
    import os, time
    os.makedirs("data", exist_ok=True)
    t0 = time.time()

    # ---- 训练集：模型只见过 3 体和 4 体系统 ----
    rng = np.random.default_rng(0)
    train = []
    for N in (3, 4):
        d = observe(rng, B=5000, N=N)
        print(f"训练 N={N}: {len(d['m'])} 个样本, "
              f"差分加速度相对真值的误差(中位数) = "
              f"{np.median(np.linalg.norm(d['a_obs']-d['a_true'],axis=-1)/np.linalg.norm(d['a_true'],axis=-1)):.2e}")
        train.append(pad(d, 4))
    train = concat(train)
    np.savez_compressed("data/train.npz", **train)

    # ---- 验证集（同分布，另一个随机种子）----
    rng = np.random.default_rng(1)
    val = concat([pad(observe(rng, B=300, N=N), 4) for N in (3, 4)])
    np.savez_compressed("data/val.npz", **val)

    # ---- 测试集：2 到 8 体，全新的随机系统 ----
    rng = np.random.default_rng(2)
    for N in range(2, 9):
        d = observe(rng, B=400, N=N, frame_stride=20)
        np.savez_compressed(f"data/test_N{N}.npz", **pad(d, 8))
        print(f"测试 N={N}: {len(d['m'])} 个样本")
    print(f"训练样本 {len(train['m'])}，验证样本 {len(val['m'])}，用时 {time.time()-t0:.0f}s")
