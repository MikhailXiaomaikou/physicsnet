"""
evaluate.py —— 网络到底学到了什么？

  1. 学到的定律：把 PhysicsNet 里的两个小网络单独拿出来，和真实定律逐点对比
  2. 物体数泛化：训练只见过 3、4 体，在 2–8 体的全新系统上测加速度误差
  3. 长时间推演：从同一初始状态出发让各模型自己往前推，看轨迹偏差和守恒量
  4. 推演示例：开普勒椭圆轨道、旋转的弹簧链、"恒星 + 行星 + 弹簧分子"6 体系统

输出：results/metrics.json 和三张图 results/fig_*.png
"""
import glob, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

import world as W
from model import build

OUT = "results"
M_LO, M_HI = W.SCENE_DEFAULTS["m_range"]
R_LO, R_HI = 0.4, 8.0                      # 训练数据覆盖的距离范围
NAMES = {"physicsnet": "PhysicsNet（牛顿骨架）", "pairwise": "PairwiseNet（只有成对+叠加）", "mlp": "黑盒 MLP（无结构）"}
SHORT = {"physicsnet": "PhysicsNet", "pairwise": "PairwiseNet", "mlp": "黑盒 MLP"}

# ---- 图表样式 ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"physicsnet": "#2a78d6", "pairwise": "#eb6834", "mlp": "#1baf7a"}
TRUTH = "#b4b2a9"


def setup_style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Heiti SC", "SimHei",
                       "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS") if f in have]
    plt.rcParams.update({
        "font.family": cjk[:1] + ["DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 10, "figure.dpi": 100, "savefig.dpi": 170,
    })


def titled(ax, title, *lines):
    """粗体标题，下面跟若干行小字说明（都在坐标区上方，互不重叠）。"""
    ax.set_title(title, pad=10 + 17 * len(lines))
    for k, line in enumerate(lines):
        ax.annotate(line, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7 + 17 * (len(lines) - 1 - k)),
                    textcoords="offset points", fontsize=9.5, color=INK2, va="bottom", ha="left")


def load_model(name, path=None):
    model = build(name)
    model.load_state_dict(torch.load(path or f"checkpoints/{name}.pt"))
    return model.double().eval()


def t64(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def model_acc_fn(model, m, K, L):
    """把模型包装成"给位置、出加速度"的函数，交给和真实世界相同的积分器。"""
    m_t, K_t, L_t = t64(m), t64(K), t64(L)

    def acc(x):
        with torch.no_grad():
            return model(t64(x), m_t, K_t, L_t).numpy()
    return acc


# ==========================================================================
# 1. 学到的定律
# ==========================================================================
@torch.no_grad()
def learned_laws(net, seed=0):
    g = torch.Generator().manual_seed(seed)
    logu = lambda lo, hi, n: torch.exp(torch.rand(n, generator=g, dtype=torch.float64) * np.log(hi / lo) + np.log(lo))
    uni = lambda lo, hi, n: torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) + lo
    res = {}

    # 规范常数 c：把所有力乘以 c、所有惯性响应除以 c，运动完全不变，所以力的"单位"
    # 无法从轨迹里确定。这里固定这一个常数（取 m·h(m) 的几何平均），其余全是网络自己学的。
    m = torch.exp(torch.linspace(np.log(M_LO), np.log(M_HI), 400, dtype=torch.float64))
    h = net.inertia(m)
    c = torch.exp((m.log() + h.log()).mean())
    p, q = np.polyfit(m.log().numpy(), (h / c).log().numpy(), 1)
    res["inertia"] = {"exponent": p, "prefactor": float(np.exp(q)),
                      "max_dev": float(((h / c) * m - 1).abs().max())}

    # 万有引力：在训练范围内随机取 (m_i, m_j, r)，没有弹簧
    n = 20000
    mi, mj, r = logu(M_LO, M_HI, n), logu(M_LO, M_HI, n), logu(R_LO, R_HI, n)
    zero = torch.zeros(n, dtype=torch.float64)
    f = c * net.pair_force(mi, mj, r, zero, zero)
    pos = f > 0
    A = torch.stack([torch.ones(n, dtype=torch.float64), mi.log(), mj.log(), r.log()], 1)[pos]
    coef = torch.linalg.lstsq(A, f[pos].log()[:, None]).solution.squeeze(1).numpy()
    ratio = (f / (W.G * mi * mj / r**2)).numpy()
    res["gravity"] = {"G": float(np.exp(coef[0])), "exp_mi": coef[1], "exp_mj": coef[2], "exp_r": coef[3],
                      "frac_attractive": float(pos.double().mean()),
                      "ratio_median": float(np.median(ratio)),
                      "abs_dev_median": float(np.median(np.abs(ratio - 1))),
                      "abs_dev_p95": float(np.percentile(np.abs(ratio - 1), 95))}
    # 弹簧读数会不会影响"引力部分"、质量会不会影响"弹簧部分"？——都不应该
    k, L, rs = uni(0.5, 4.0, n), uni(0.5, 2.5, n), uni(0.5, 3.5, n)
    spring = c * (net.pair_force(mi, mj, rs, k, L) - net.pair_force(mi, mj, rs, zero, zero))
    true = k * (rs - L)
    B = torch.stack([k * rs, k * L, k, rs, L, torch.ones(n, dtype=torch.float64)], 1)
    sc = torch.linalg.lstsq(B, spring[:, None]).solution.squeeze(1).numpy()
    m2i, m2j = logu(M_LO, M_HI, n), logu(M_LO, M_HI, n)       # 同样的弹簧，换一对质量
    spring2 = c * (net.pair_force(m2i, m2j, rs, k, L) - net.pair_force(m2i, m2j, rs, zero, zero))
    res["spring"] = {"coef_k_r": sc[0], "coef_k_L": sc[1], "other_coefs_max": float(np.abs(sc[2:]).max()),
                     "rel_rms_dev": float(((spring - true).pow(2).mean() / true.pow(2).mean()).sqrt()),
                     "mass_sensitivity": float(((spring - spring2).pow(2).mean() / true.pow(2).mean()).sqrt())}

    # 画图用的曲线
    curves = {"c": float(c)}
    rr = torch.exp(torch.linspace(np.log(R_LO), np.log(R_HI), 16, dtype=torch.float64))
    curves["gravity"] = []
    for a, b in [(3.0, 3.0), (0.5, 2.0), (0.3, 0.3)]:
        fa = c * net.pair_force(torch.full_like(rr, a), torch.full_like(rr, b), rr, torch.zeros_like(rr), torch.zeros_like(rr))
        curves["gravity"].append({"mi": a, "mj": b, "r": rr.numpy(), "f": fa.numpy()})
    rl = torch.linspace(0.5, 3.5, 16, dtype=torch.float64)
    curves["spring"] = []
    one = torch.ones_like(rl)
    for kk, LL in [(4.0, 1.5), (2.0, 1.0), (1.0, 2.2)]:
        s = c * (net.pair_force(one, one, rl, kk * one, LL * one) - net.pair_force(one, one, rl, 0 * one, 0 * one))
        curves["spring"].append({"k": kk, "L": LL, "r": rl.numpy(), "f": s.numpy()})
    mm = torch.exp(torch.linspace(np.log(M_LO), np.log(M_HI), 16, dtype=torch.float64))
    curves["inertia"] = {"m": mm.numpy(), "h": (net.inertia(mm) / c).numpy()}
    return res, curves


@torch.no_grad()
def extrapolation(net, c):
    """训练范围之外会怎样？返回 模型值/真实值（1 表示完全正确）。
    训练范围：距离 0.4–8，质量 0.25–4，劲度 0.5–4。"""
    T = lambda v: torch.tensor([float(v)], dtype=torch.float64)
    z = T(0)
    out = {"r": {}, "m": {}, "k": {}}
    for r in (0.1, 0.2, 0.3, 0.4, 2.0, 8.0, 10.0, 16.0, 32.0):          # 引力，m1 = m2 = 1
        out["r"][r] = float(c * net.pair_force(T(1), T(1), T(r), z, z)) * r * r
    for m in (0.1, 0.2, 0.25, 1.0, 4.0, 5.0, 8.0, 16.0):                # 引力（r = 2）和惯性
        out["m"][m] = {"force": float(c * net.pair_force(T(m), T(m), T(2), z, z)) * 4 / m**2,
                       "inertia": float(net.inertia(T(m)) / c) * m}
    for k in (0.25, 0.5, 2.0, 4.0, 6.0, 8.0):                           # 弹簧，L = 1，r = 2
        sp = float(c * (net.pair_force(T(1), T(1), T(2), T(k), T(1)) - net.pair_force(T(1), T(1), T(2), z, z)))
        out["k"][k] = sp / k
    return out


def fig_laws(res, curves):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    fig.subplots_adjust(left=0.055, right=0.975, top=0.74, bottom=0.115, wspace=0.27)
    dot = dict(marker="o", ms=6.5, mfc=COLOR["physicsnet"], mec=SURFACE, mew=1.2, ls="none", zorder=3)

    ax = axes[0]
    g = res["gravity"]
    for cv in curves["gravity"]:
        rf = np.geomspace(R_LO, R_HI, 100)
        ax.plot(rf, W.G * cv["mi"] * cv["mj"] / rf**2, color=TRUTH, lw=3)
        ax.plot(cv["r"], cv["f"], **dot)
        ax.annotate(f"$m_1$={cv['mi']:g}, $m_2$={cv['mj']:g}", (cv["r"][-1], cv["f"][-1]), xytext=(9, 0),
                    textcoords="offset points", fontsize=9.5, color=INK2, ha="left", va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.35, 27)        # 右侧留白给曲线末端的标注
    ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"]); ax.minorticks_off()
    ax.set_xlabel("两物体的距离 r"); ax.set_ylabel("成对吸引力 F")
    titled(ax, "万有引力定律",
           f"学到的:  $F = {g['G']:.3f}\\; m_1^{{\\,{g['exp_mi']:.3f}}}\\; m_2^{{\\,{g['exp_mj']:.3f}}}\\; r^{{\\,{g['exp_r']:.3f}}}$",
           "真实的:  $F = 1 \\cdot m_1\\, m_2\\, /\\, r^{2}$")

    ax = axes[1]
    s = res["spring"]
    ax.axhline(0, color=AXIS, lw=1)
    for cv in curves["spring"]:
        rf = np.linspace(0.5, 3.5, 50)
        ax.plot(rf, cv["k"] * (rf - cv["L"]), color=TRUTH, lw=3)
        ax.plot(cv["r"], cv["f"], **dot)
        ax.annotate(f"k={cv['k']:g}, L={cv['L']:g}", (cv["r"][-1], cv["f"][-1]), xytext=(9, 0),
                    textcoords="offset points", fontsize=9.5, color=INK2, ha="left", va="center")
    ax.set_xlim(0.38, 4.5); ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_xlabel("弹簧两端的距离 r"); ax.set_ylabel("弹簧带来的那部分力（正 = 往回拉）")
    titled(ax, "胡克定律",
           f"学到的:  $F = {s['coef_k_r']:.3f}\\, k\\, r - {-s['coef_k_L']:.3f}\\, k\\, L$",
           "真实的:  $F = k\\,(r - L)$")

    ax = axes[2]
    i = res["inertia"]
    mf = np.geomspace(M_LO, M_HI, 100)
    ax.plot(mf, 1 / mf, color=TRUTH, lw=3)
    ax.plot(curves["inertia"]["m"], curves["inertia"]["h"], **dot)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("物体的质量读数 m"); ax.set_ylabel("单位合力产生的加速度")
    titled(ax, "质量如何决定加速度（第二定律）",
           f"学到的:  $a = F \\cdot m^{{\\,{i['exponent']:.3f}}}$",
           "真实的:  $a = F\\, /\\, m$")

    handles = [Line2D([], [], color=TRUTH, lw=3, label="真实定律（网络从未见过公式）"),
               Line2D([], [], label="PhysicsNet 只从运动轨迹中学到的", **dot)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.995), ncol=2, handlelength=2.2, columnspacing=2.5)
    fig.savefig(f"{OUT}/fig_laws.png")
    plt.close(fig)


# ==========================================================================
# 2. 物体数泛化（单步加速度误差）
# ==========================================================================
@torch.no_grad()
def generalization(models):
    out = {name: {} for name in models}
    for N in range(2, 9):
        d = np.load(f"data/test_N{N}.npz")
        x, m, K, L, a_true = (t64(d[k][:, :N]) for k in ("x", "m", "K", "L", "a_true"))
        K, L = K[:, :, :N], L[:, :, :N]
        for name, model in models.items():
            a = model(x, m, K, L)
            err, ref = (a - a_true).norm(dim=-1), a_true.norm(dim=-1)
            out[name][N] = {"median_rel_err": float((err / ref).median()),
                            "rel_rms": float((err.pow(2).sum() / ref.pow(2).sum()).sqrt()),
                            "n_samples": int(len(x))}
    return out


# ==========================================================================
# 3. 长时间推演
# ==========================================================================
def stable_systems(rng, N, n_want, T, dt, record_every, batch=1500, max_rounds=8):
    """抽取在真实物理下全程都处于训练距离范围内的随机系统（只依据真实轨迹筛选，与模型无关）。"""
    got, tried = [], 0
    for _ in range(max_rounds):
        x0, v0, m, K, L = W.sample_scene(rng, batch, N)
        xs, vs, alive = W.simulate(x0, v0, lambda x: W.accelerations(x, m, K, L), T, dt, record_every)
        pd = np.stack([W.pair_distances(xs[:, t]) for t in range(xs.shape[1])], 1)
        ok = alive[:, -1] & (pd.min(axis=(1, 2)) >= R_LO) & (pd.max(axis=(1, 2)) <= R_HI)
        got.append([a[ok] for a in (x0, v0, m, K, L, xs, vs)])
        tried += batch
        if sum(len(g[0]) for g in got) >= n_want:
            break
    cat = [np.concatenate([g[i] for g in got])[:n_want] for i in range(7)]
    return cat, tried


def conserved_errors(xs, vs, m, K, L):
    """沿一条轨迹，用"真实"的能量/动量/角动量公式去量它的守恒程度。返回 (B, 帧数)。"""
    B, F = xs.shape[:2]
    E = np.stack([W.energy(xs[:, t], vs[:, t], m, K, L) for t in range(F)], 1)
    P = np.stack([W.momentum(vs[:, t], m) for t in range(F)], 1)
    Lz = np.stack([W.angular_momentum(xs[:, t], vs[:, t], m) for t in range(F)], 1)
    kin0 = 0.5 * (m * (vs[:, 0] ** 2).sum(-1)).sum(-1)
    iu = np.triu_indices(m.shape[1], 1)
    r0 = W.pair_distances(xs[:, 0])
    pot0 = (W.G * (m[:, iu[0]] * m[:, iu[1]]) / r0 + 0.5 * K[:, iu[0], iu[1]] * (r0 - L[:, iu[0], iu[1]]) ** 2).sum(-1)
    p_scale = (m * np.linalg.norm(vs[:, 0], axis=-1)).sum(-1)
    l_scale = (m * np.linalg.norm(xs[:, 0], axis=-1) * np.linalg.norm(vs[:, 0], axis=-1)).sum(-1)
    return {"energy": np.abs(E - E[:, :1]) / (kin0 + pot0)[:, None],
            "momentum": np.linalg.norm(P - P[:, :1], axis=-1) / p_scale[:, None],
            "angular_momentum": np.abs(Lz - Lz[:, :1]) / l_scale[:, None]}


def rollout_stats(models, N, n_sys=60, T=10.0, dt=4e-3, record_every=10, seed=100):
    rng = np.random.default_rng(seed + N)
    (x0, v0, m, K, L, xs_true, vs_true), tried = stable_systems(rng, N, n_sys, T, dt, record_every)
    size = np.sqrt((np.linalg.norm(x0 - x0.mean(1, keepdims=True), axis=-1) ** 2).mean(-1))   # 系统尺度
    t = np.arange(xs_true.shape[1]) * dt * record_every
    out = {"t": t, "n_systems": len(x0), "n_candidates": tried, "models": {}}
    out["truth_energy_err"] = np.median(conserved_errors(xs_true, vs_true, m, K, L)["energy"], 0)
    for name, model in models.items():
        xs, vs, alive = W.simulate(x0, v0, model_acc_fn(model, m, K, L), T, dt, record_every)
        cons = conserved_errors(xs, vs, m, K, L)
        pos = np.sqrt((np.linalg.norm(xs - xs_true, axis=-1) ** 2).mean(-1)) / size[:, None]
        out["models"][name] = {"position": np.median(pos, 0), "position_p90": np.percentile(pos, 90, axis=0),
                               **{k: np.median(v, 0) for k, v in cons.items()},
                               "energy_p90": np.percentile(cons["energy"], 90, axis=0),
                               "crashed_frac": float((~alive[:, -1]).mean())}
    return out


def fig_generalization(gen, roll):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.8))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.855, bottom=0.065, wspace=0.24, hspace=0.36)
    order = ["physicsnet", "pairwise", "mlp"]

    def end_label(ax, name, x, y, above):
        ax.annotate(SHORT[name], (x, y), xytext=(0, 7 if above else -11), textcoords="offset points",
                    fontsize=9.5, color=INK2, ha="right", va="bottom" if above else "top")

    ax = axes[0, 0]
    ax.axvspan(2.75, 4.25, color="#f0efec", lw=0)
    ax.text(3.5, 0.985, "训练时见过", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5, color=INK2)
    for name in order:
        Ns = sorted(gen[name]); y = [100 * gen[name][N]["median_rel_err"] for N in Ns]
        ax.plot(Ns, y, color=COLOR[name], marker="o", ms=7, mec=SURFACE, mew=1.5)
        end_label(ax, name, Ns[-1], y[-1], above=(name != "mlp"))
    ax.set_yscale("log"); ax.set_xlabel("系统里的物体个数"); ax.set_ylabel("加速度相对误差（中位数，%）")
    ax.set_title("换一个物体数还准不准？（全新的随机系统）")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    t = roll["t"]
    panels = [(axes[0, 1], "position", "轨迹偏差 / 系统尺度（中位数）", "自己往前推演：偏离真实轨迹多少？"),
              (axes[1, 0], "energy", "能量相对变化（中位数）", "推演过程中能量守恒吗？"),
              (axes[1, 1], "momentum", "总动量相对变化（中位数）", "推演过程中动量守恒吗？")]
    for ax, key, ylabel, title in panels:
        for name in order:
            y = np.maximum(roll["models"][name][key], 1e-16)
            ax.plot(t[1:], y[1:], color=COLOR[name])
            end_label(ax, name, t[-1], y[-1], above=False)
        ax.set_yscale("log"); ax.set_xlabel("推演时间"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xlim(0, t[-1])
    handles = [Line2D([], [], color=COLOR[n], lw=2.5, marker="o", ms=7, mec=SURFACE, mew=1.5, label=NAMES[n]) for n in order]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.995), ncol=3, columnspacing=2.5)
    fig.text(0.075, 0.928, f"三个模型用同样的观测数据、同样的训练步数。后三幅图: {roll['n_systems']} 个随机 {roll['N']} 体系统，"
             f"从相同的初始状态出发各自推演 {t[-1]:g} 个时间单位。", fontsize=9.5, color=INK2)
    fig.savefig(f"{OUT}/fig_generalization.png")
    plt.close(fig)


# ==========================================================================
# 4. 推演示例（三个手工搭的场景，训练集中没有这样的系统）
# ==========================================================================
def scene_kepler():
    """两体椭圆轨道（训练时从未见过 2 体系统）。半长轴 2，偏心率 0.6。"""
    m = np.array([[3.0, 0.5]]); M = m.sum()
    a, e = 2.0, 0.6
    r_apo = a * (1 + e); v_apo = np.sqrt(W.G * M * (1 - e) / (a * (1 + e)))
    x = np.array([[[0.0, 0.0], [r_apo, 0.0]]]); v = np.array([[[0.0, 0.0], [0.0, v_apo]]])
    period = 2 * np.pi * np.sqrt(a**3 / (W.G * M))
    return _to_com(x, v, m, np.zeros((1, 2, 2)), np.zeros((1, 2, 2))), 4 * period


def scene_chain():
    """五个质点用四根弹簧串成一条链，一边整体旋转一边振动、弯曲。"""
    N = 5
    m = np.array([[0.5, 0.3, 0.6, 0.3, 0.5]])
    x = np.zeros((1, N, 2)); x[0, :, 0] = np.arange(N) * 1.0 - 2.0
    x[0, :, 1] = np.array([0.0, 0.25, -0.1, 0.25, 0.0])
    v = np.zeros((1, N, 2)); v[0, :, 1] = 0.55 * x[0, :, 0]; v[0, :, 0] = np.array([-0.3, 0.2, 0.0, -0.2, 0.3])
    K = np.zeros((1, N, N)); L = np.zeros((1, N, N))
    for i in range(N - 1):
        K[0, i, i + 1] = K[0, i + 1, i] = 3.0; L[0, i, i + 1] = L[0, i + 1, i] = 1.0
    return _to_com(x, v, m, K, L), 20.0


def scene_six_body():
    """恒星 + 一颗近轨行星 + 一个由 4 个质点和 6 根弹簧组成、自转着的"分子"绕恒星公转。"""
    m = np.array([[3.6, 0.25, 0.25, 0.25, 0.25, 0.25]])
    N = 6
    x = np.zeros((1, N, 2)); v = np.zeros((1, N, 2))
    x[0, 1] = [0.7, 0.0]; v[0, 1] = [0.0, np.sqrt(W.G * 3.6 / 0.7)]
    R, side, spin = 3.8, 0.7, 1.2
    center = np.array([-R, 0.0]); v_orb = np.array([0.0, -np.sqrt(W.G * (3.6 + 0.25) / R)])
    corners = 0.5 * side * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    for q in range(4):
        x[0, 2 + q] = center + corners[q]
        v[0, 2 + q] = v_orb + spin * np.array([-corners[q, 1], corners[q, 0]])
    K = np.zeros((1, N, N)); L = np.zeros((1, N, N))
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = 2 + a, 2 + b
            K[0, i, j] = K[0, j, i] = 3.0
            L[0, i, j] = L[0, j, i] = np.linalg.norm(corners[a] - corners[b])
    return _to_com(x, v, m, K, L), 24.0


def _to_com(x, v, m, K, L):
    x = x - (m[..., None] * x).sum(1, keepdims=True) / m.sum(1)[:, None, None]
    v = v - (m[..., None] * v).sum(1, keepdims=True) / m.sum(1)[:, None, None]
    return x, v, m, K, L


def precession_per_orbit(xs, vs, m, t):
    """两体轨道的椭圆长轴方向每圈转过多少度。严格的平方反比力下椭圆不转（0°/圈），
    所以它是检验"力是否真的按 1/r² 变化"的一把很灵敏的尺子。
    做法：逐帧计算拉普拉斯–龙格–楞次矢量的方向，对时间做线性拟合。"""
    r = xs[0, :, 1] - xs[0, :, 0]
    v = vs[0, :, 1] - vs[0, :, 0]
    mu = W.G * m.sum()
    Lz = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    rn = np.linalg.norm(r, axis=-1)
    ex = v[:, 1] * Lz / mu - r[:, 0] / rn
    ey = -v[:, 0] * Lz / mu - r[:, 1] / rn
    ang = np.unwrap(np.arctan2(ey, ex))
    energy = 0.5 * (v**2).sum(-1) - mu / rn
    period = 2 * np.pi * mu / (-2 * energy.mean()) ** 1.5
    return float(np.degrees(np.polyfit(t, ang, 1)[0] * period))


def showcase(net, dt=4e-3, record_every=5):
    scenes = {"kepler": scene_kepler(), "chain": scene_chain(), "six_body": scene_six_body()}
    out = {}
    for key, ((x, v, m, K, L), T) in scenes.items():
        xs_t, vs_t, _ = W.simulate(x, v, lambda y: W.accelerations(y, m, K, L), T, dt, record_every)
        xs_m, vs_m, alive = W.simulate(x, v, model_acc_fn(net, m, K, L), T, dt, record_every)
        pd = np.stack([W.pair_distances(xs_t[:, t]) for t in range(xs_t.shape[1])], 1)
        size = np.sqrt((np.linalg.norm(x - x.mean(1, keepdims=True), axis=-1) ** 2).mean())
        pos = np.sqrt((np.linalg.norm(xs_m - xs_t, axis=-1) ** 2).mean(-1))[0] / size
        out[key] = {"T": T, "truth": xs_t[0], "model": xs_m[0], "K": K[0], "m": m[0],
                    "r_range": [float(pd.min()), float(pd.max())],
                    "final_pos_err": float(pos[-1]), "max_pos_err": float(pos.max()),
                    "energy_err_max": float(conserved_errors(xs_m, vs_m, m, K, L)["energy"].max()),
                    "survived": bool(alive[0, -1])}
        if key == "kepler":
            t = np.arange(xs_t.shape[1]) * dt * record_every
            out[key]["precession_deg_per_orbit"] = precession_per_orbit(xs_m, vs_m, m, t)
            out[key]["precession_truth"] = precession_per_orbit(xs_t, vs_t, m, t)
    return out


def fig_rollouts(show):
    titles = {"kepler": ("两体椭圆轨道（训练时没见过 2 体）", f"推演 4 圈 · 椭圆长轴进动 {show['kepler']['precession_deg_per_orbit']:+.2f}°/圈（真实为 0）"),
              "chain": ("旋转并振动的弹簧链（5 体）", "推演 20 个时间单位"),
              "six_body": ("恒星 + 行星 + 弹簧“分子”（6 体）", "行星绕恒星约 14 圈，分子公转约 1 圈")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.775, bottom=0.03, wspace=0.06)
    for ax, key in zip(axes, ("kepler", "chain", "six_body")):
        s = show[key]
        tr, mo = s["truth"], s["model"]
        for i in range(tr.shape[1]):
            ax.plot(tr[:, i, 0], tr[:, i, 1], color=TRUTH, lw=3.6, solid_capstyle="round")
        for i in range(tr.shape[1]):
            ax.plot(mo[:, i, 0], mo[:, i, 1], color=COLOR["physicsnet"], lw=1.1)
        iu = np.triu_indices(tr.shape[1], 1)
        for i, j in zip(*iu):                                   # 终点时刻的弹簧
            if s["K"][i, j] > 0:
                ax.plot(mo[-1, [i, j], 0], mo[-1, [i, j], 1], color=INK2, lw=1.0, zorder=4)
        size = 30 + 55 * np.sqrt(s["m"])
        ax.scatter(tr[-1, :, 0], tr[-1, :, 1], s=size * 1.9, facecolor="none", edgecolor=INK2, lw=1.3, zorder=5)
        ax.scatter(mo[-1, :, 0], mo[-1, :, 1], s=size * 0.75, color=COLOR["physicsnet"], edgecolor=SURFACE, lw=1.2, zorder=6)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.margins(0.06)
        titled(ax, titles[key][0], titles[key][1], f"终点位置偏差 = 系统尺度的 {100*s['final_pos_err']:.1f}%")
    handles = [Line2D([], [], color=TRUTH, lw=3.6, label="真实轨迹"),
               Line2D([], [], color=COLOR["physicsnet"], lw=1.3, label="PhysicsNet 自己推演的轨迹"),
               Line2D([], [], marker="o", ms=10, mfc="none", mec=INK2, mew=1.3, ls="none", label="真实终点位置"),
               Line2D([], [], marker="o", ms=7, mfc=COLOR["physicsnet"], mec=SURFACE, ls="none", label="PhysicsNet 终点位置")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.99), ncol=4, columnspacing=2.5)
    fig.savefig(f"{OUT}/fig_rollouts.png")
    plt.close(fig)


# ==========================================================================
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    setup_style()
    models = {n: load_model(n) for n in ("physicsnet", "pairwise", "mlp") if os.path.exists(f"checkpoints/{n}.pt")}
    net = models["physicsnet"]

    laws, curves = learned_laws(net)
    fig_laws(laws, curves)
    g, s, i = laws["gravity"], laws["spring"], laws["inertia"]
    print("== 学到的定律 ==")
    print(f"  引力:  F = {g['G']:.4f} · m_i^{g['exp_mi']:.4f} · m_j^{g['exp_mj']:.4f} · r^{g['exp_r']:.4f}"
          f"   (逐点偏离真值: 中位 {g['abs_dev_median']:.2%}, 95 分位 {g['abs_dev_p95']:.2%})")
    print(f"  弹簧:  F = {s['coef_k_r']:.4f}·k·r {s['coef_k_L']:+.4f}·k·L   (其余项系数最大 {s['other_coefs_max']:.4f}; "
          f"相对真值的 RMS 偏差 {s['rel_rms_dev']:.2%}; 换质量引起的变化 {s['mass_sensitivity']:.2%})")
    print(f"  惯性:  a = F · {i['prefactor']:.4f} · m^{i['exponent']:.4f}   (最大偏差 {i['max_dev']:.2%})")

    seeds = {}
    for path in sorted(glob.glob("checkpoints/physicsnet_seed*.pt")):           # 换随机种子重新训练的结果（如果有）
        sl, _ = learned_laws(load_model("physicsnet", path))
        seeds[os.path.basename(path)] = sl
        print(f"  [{os.path.basename(path)}] 引力 r^{sl['gravity']['exp_r']:.4f}, m^{sl['gravity']['exp_mi']:.4f}, G={sl['gravity']['G']:.4f}; "
              f"弹簧 {sl['spring']['coef_k_r']:.4f}/{sl['spring']['coef_k_L']:.4f}; 惯性 m^{sl['inertia']['exponent']:.4f}")

    extra = extrapolation(net, curves["c"])
    print("== 训练范围之外（模型值 / 真实值）==")
    print("  引力随距离: " + "  ".join(f"r={r:g}: {v:.3f}" for r, v in extra["r"].items()))
    print("  引力随质量: " + "  ".join(f"m={m:g}: {v['force']:.3f}" for m, v in extra["m"].items()))
    print("  惯性随质量: " + "  ".join(f"m={m:g}: {v['inertia']:.3f}" for m, v in extra["m"].items()))
    print("  弹簧随劲度: " + "  ".join(f"k={k:g}: {v:.3f}" for k, v in extra["k"].items()))

    gen = generalization(models)
    print("== 物体数泛化：加速度相对误差中位数 ==")
    for name in models:
        print(f"  {name:10s}", "  ".join(f"N={N}: {gen[name][N]['median_rel_err']:.2%}" for N in sorted(gen[name])))

    # 随机系统里引力很强，绝大多数很快就会出现近距离交会（超出训练见过的距离范围），
    # 所以只能测"真实轨迹全程留在训练范围内"的那一小部分系统：3 体测 10 个时间单位，4 体测 5 个。
    rolls = {}
    for N, T in ((3, 10.0), (4, 5.0)):
        rolls[N] = rollout_stats(models, N, T=T); rolls[N]["N"] = N
        r = rolls[N]
        print(f"== 推演 {r['n_systems']} 个随机 {N} 体系统（从 {r['n_candidates']} 个候选中筛出），时长 {r['t'][-1]:g} ==")
        print(f"  （参照：真实物理用同一积分器推演，能量变化 {r['truth_energy_err'][-1]:.1e}）")
        for name in models:
            mm = r["models"][name]
            print(f"  {name:10s} 终点轨迹偏差 {mm['position'][-1]:.2%}  能量变化 {mm['energy'][-1]:.2e}  "
                  f"动量变化 {mm['momentum'][-1]:.2e}  角动量变化 {mm['angular_momentum'][-1]:.2e}  相撞比例 {mm['crashed_frac']:.0%}")
    fig_generalization(gen, rolls[3])

    show = showcase(net)
    fig_rollouts(show)
    print("== 推演示例 (PhysicsNet) ==")
    for key, sc in show.items():
        note = f"  椭圆长轴进动 {sc['precession_deg_per_orbit']:+.3f}°/圈" if key == "kepler" else ""
        print(f"  {key:9s} 距离范围 {sc['r_range'][0]:.2f}–{sc['r_range'][1]:.2f}  终点偏差 {sc['final_pos_err']:.2%}  "
              f"最大能量变化 {sc['energy_err_max']:.2e}{note}")

    for sc in show.values():
        for k in ("truth", "model", "K", "m"):
            sc.pop(k)
    json.dump(jsonable({"laws": laws, "laws_other_seeds": seeds, "extrapolation": extra,
                        "generalization": gen, "rollouts": rolls, "showcase": show}),
              open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
    print(f"已写入 {OUT}/metrics.json 和 {OUT}/fig_*.png")


if __name__ == "__main__":
    main()
