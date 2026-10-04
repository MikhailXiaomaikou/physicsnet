"""
model.py —— 三个模型，物理结构由少到多

  BlackBoxMLP  没有任何物理结构：整个系统拉平成一个向量，直接输出所有加速度。
  PairwiseNet  只知道"作用是成对的、可以叠加"：a_i = Σ_j ψ(i, j)。
  PhysicsNet   当前阶段（v0）只覆盖牛顿力学：把它的骨架做进结构，具体定律留给网络去学。

PhysicsNet 目前写死在结构里的假设（牛顿力学的骨架）：
  1. 力是成对的：每一对物体之间有一个相互作用；
  2. 作用力与反作用力大小相等、方向相反，沿两者连线（第三定律的强形式）；
  3. 力的大小只取决于两者的距离和属性（与绝对位置、朝向、速度无关）；
  4. 叠加原理：一个物体受的合力 = 各成对力的矢量和；
  5. 加速度 = 合力 × 一个只和该物体自身质量读数有关的响应系数。

PhysicsNet 需要从数据里学的东西（两个小网络）：
  force_net   φ(m_i, m_j, r, 有无弹簧, k, L) → 成对力的大小    （万有引力定律、胡克定律）
  inertia_net h(m)                          → 惯性响应系数    （第二定律里的 1/m）

输入给网络时，m 和 r 同时提供原值和对数值（只是换一种刻度，不含任何定律信息）。
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, depth=3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _amp_exp(o):
    """输出层：前几个分量是带符号的"幅度"，最后一个分量是"指数"，结果 = 幅度 × exp(指数)。
    力的大小跨越好几个数量级（远处的引力很弱，近处很强），这样的输出形式能让网络在
    各个量级上都保持相对精度。它本身不包含任何具体定律。
    （两个对照模型用的是普通线性输出：实测这种输出形式对它们反而更差——训练会卡住。）"""
    return o[..., :-1] * torch.exp(o[..., -1:])


def _pair_geometry(x, mask):
    """d[b,i,j] = x_j − x_i，r = |d|，pair_ok = 这一对是否真实存在（i≠j 且两个槽位都有物体）。"""
    N = x.shape[1]
    d = x[:, None, :, :] - x[:, :, None, :]
    eye = torch.eye(N, dtype=torch.bool, device=x.device)
    pair_ok = (~eye)[None] & mask[:, :, None] & mask[:, None, :]
    r2 = (d**2).sum(-1)
    r = torch.sqrt(torch.where(pair_ok, r2, torch.ones_like(r2)))
    return d, r, pair_ok


class PhysicsNet(nn.Module):
    def __init__(self, hidden=128, depth=3, log_features=True):
        super().__init__()
        self.log_features = log_features
        self.force_net = MLP(9 if log_features else 6, 2, hidden, depth)
        self.inertia_net = MLP(2 if log_features else 1, 1, 32, 2)

    # ---- 学到的"定律"：这两个函数可以单独拿出来查看 ----
    def _phi(self, mi, mj, r, k, L):
        feats = [mi, mj, r, (k > 0).to(r.dtype), k, L]
        if self.log_features:
            feats += [mi.log(), mj.log(), r.log()]
        return _amp_exp(self.force_net(torch.stack(feats, dim=-1))).squeeze(-1)

    def pair_force(self, mi, mj, r, k, L):
        """学到的成对吸引力大小（正 = 相吸）。k=0 表示这一对之间没有弹簧。
        对两个物体交换对称，所以 i 对 j 的力必然等于 j 对 i 的力的反向。"""
        return 0.5 * (self._phi(mi, mj, r, k, L) + self._phi(mj, mi, r, k, L))

    def inertia(self, m):
        """学到的惯性响应 h(m)：加速度 = 合力 × h(m)。"""
        feats = [m, m.log()] if self.log_features else [m]
        return torch.exp(self.inertia_net(torch.stack(feats, dim=-1)).squeeze(-1))

    # ---- 骨架：成对力 → 叠加 → 加速度 ----
    def net_force(self, x, m, K, L, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        B, N, _ = x.shape
        d, r, pair_ok = _pair_geometry(x, mask)
        b, i, j = pair_ok.nonzero(as_tuple=True)              # 所有真实存在的有向对 i←j
        phi = x.new_zeros(B, N, N)
        phi[b, i, j] = self._phi(m[b, i], m[b, j], r[b, i, j], K[b, i, j], L[b, i, j])
        f = 0.5 * (phi + phi.transpose(1, 2))                 # 交换对称 ⇒ 作用力 = −反作用力
        F = (f[..., None] * d / r[..., None]).sum(dim=2)      # 沿连线方向，矢量叠加
        return F, mask

    def forward(self, x, m, K, L, mask=None):
        F, mask = self.net_force(x, m, K, L, mask)
        m_safe = torch.where(mask, m, torch.ones_like(m))
        return F * self.inertia(m_safe)[..., None] * mask[..., None]


class PairwiseNet(nn.Module):
    """只有"成对 + 叠加"：a_i = Σ_j ψ(m_i, m_j, x_j − x_i, 有无弹簧, k, L)。
    ψ 直接输出加速度矢量；不保证作用力=反作用力，不保证力沿连线，也没有旋转不变性。"""

    def __init__(self, D=2, hidden=128, depth=3):
        super().__init__()
        self.net = MLP(9 + D, D, hidden, depth)

    def forward(self, x, m, K, L, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        B, N, D = x.shape
        d, r, pair_ok = _pair_geometry(x, mask)
        b, i, j = pair_ok.nonzero(as_tuple=True)
        mi, mj, rr, k, l = m[b, i], m[b, j], r[b, i, j], K[b, i, j], L[b, i, j]
        feats = torch.cat([torch.stack([mi, mj, rr, (k > 0).to(x.dtype), k, l,
                                        mi.log(), mj.log(), rr.log()], dim=-1), d[b, i, j]], dim=-1)
        contrib = x.new_zeros(B, N, N, D)
        contrib[b, i, j] = self.net(feats)
        return contrib.sum(dim=2)


class BlackBoxMLP(nn.Module):
    """没有任何物理结构的对照。输入：所有物体的位置（相对几何中心）、质量、
    以及每一对的弹簧读数，拼成一个长向量；输出：所有物体的加速度。"""

    def __init__(self, n_max=8, D=2, hidden=256, depth=4):
        super().__init__()
        self.n_max, self.D = n_max, D
        iu = torch.triu_indices(n_max, n_max, 1)
        self.register_buffer("iu", iu)
        n_pair = iu.shape[1]
        self.net = MLP(n_max * D + 2 * n_max + 3 * n_pair, n_max * D, hidden, depth)

    def forward(self, x, m, K, L, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        B, N, D = x.shape
        P = self.n_max
        xp = x.new_zeros(B, P, D); xp[:, :N] = x
        mp = x.new_zeros(B, P); mp[:, :N] = m * mask
        kp = x.new_zeros(B, P, P); kp[:, :N, :N] = K
        lp = x.new_zeros(B, P, P); lp[:, :N, :N] = L
        maskp = torch.zeros(B, P, dtype=torch.bool, device=x.device); maskp[:, :N] = mask
        w = maskp.to(x.dtype)
        center = (xp * w[..., None]).sum(1, keepdim=True) / w.sum(1)[:, None, None]
        xc = (xp - center) * w[..., None]
        ku = kp[:, self.iu[0], self.iu[1]]
        lu = lp[:, self.iu[0], self.iu[1]]
        feats = torch.cat([xc.flatten(1), mp, w, (ku > 0).to(x.dtype), ku, lu], dim=1)
        a = self.net(feats).view(B, P, D) * w[..., None]
        return a[:, :N]


def build(name):
    return {"physicsnet": PhysicsNet, "pairwise": PairwiseNet, "mlp": BlackBoxMLP}[name]()
