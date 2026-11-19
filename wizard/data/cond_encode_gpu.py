"""Proxy-video encoder on the GPU (torch): the same encoding as cond_encode.py (numpy reference), ~10x faster per window.
All per-frame arrays are batched [T,H,W]; cameras: pos [T,3], R [T,3,3] rows right/up/forward (y up), f, cx, cy (scalars)."""
import math, torch, torch.nn.functional as F
L = torch.tensor([0.45, 0.84, -0.30]); L = L / L.norm()
G_UNK, G_COARSE, G_EX0, G_EX1 = 38, 77, 153, 255

def rays(T, H, W, f, cx, cy, R, dev):
    """[T,H,W,3] world directions with forward component 1 (planar depth z -> point = pos + z * ray)"""
    v, u = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    a = ((u - cx) / f)[None, ..., None]; b = ((v - cy) / f)[None, ..., None]
    return a * R[:, None, None, 0] - b * R[:, None, None, 1] + R[:, None, None, 2]

def gblur(x, sig):
    """separable gaussian over the last two dims of [T,H,W]"""
    if sig <= 0.3: return x
    r = int(math.ceil(3 * sig)); k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sig * sig)); k = k / k.sum()
    y = F.conv2d(F.pad(x[:, None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1)); return y[:, 0]

def pushpull(val, w, levels=8):
    """fill where w == 0 from the surroundings (pull-push pyramid): val, w [T,H,W]"""
    pyr = [(val * w, w)]
    for _ in range(levels):
        vw, ww = pyr[-1]
        if min(vw.shape[-2:]) < 4: break
        pyr.append((F.avg_pool2d(vw[:, None], 2, ceil_mode=True)[:, 0], F.avg_pool2d(ww[:, None], 2, ceil_mode=True)[:, 0]))
    vw, ww = pyr[-1]; cur = vw / ww.clamp_min(1e-6)
    for vw, ww in reversed(pyr[:-1]):
        up = F.interpolate(cur[:, None], size=vw.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        a = ww.clamp(0, 1); cur = a * (vw / ww.clamp_min(1e-6)) + (1 - a) * up
    return cur

def temporal_fill(Z, valid, CH, pos, R, f, cx, cy, nb=(-60, -40, -25, -12, 12, 25, 40, 60), step=2):
    """depth behind the characters from other frames reprojected (z-buffer min); returns Z filled, still-missing mask"""
    T, H, W = Z.shape; dev = Z.device; ray = rays(T, H, W, f, cx, cy, R, dev)[:, ::step, ::step]
    P = pos[:, None, None] + Z[:, ::step, ::step, None] * ray; bg = (valid & ~CH)[:, ::step, ::step]
    need = CH & valid; out = Z.clone(); got = torch.zeros_like(need)
    for t in range(T):
        if not need[t].any(): continue
        buf = torch.full((H * W,), float("inf"), device=dev)
        for d in nb:
            s = t + d
            if not (0 <= s < T): continue
            pts = P[s][bg[s]]; rel = pts - pos[t]; zc = rel @ R[t, 2]; k = zc > 0.1; rel, zc = rel[k], zc[k]
            u = torch.floor(cx + f * (rel @ R[t, 0]) / zc).long(); v = torch.floor(cy - f * (rel @ R[t, 1]) / zc).long()
            k = (u >= 0) & (u < W) & (v >= 0) & (v < H); u, v, zc = u[k], v[k], zc[k]; k = need[t][v, u]
            if k.any(): buf.scatter_reduce_(0, v[k] * W + u[k], zc[k], reduce="amin")
        b = buf.view(H, W); b = torch.where(torch.isfinite(b), b, torch.full_like(b, 1e9))
        b = -F.max_pool2d(-b[None, None], 2 * step + 1, 1, step)[0, 0]; g = (b < 1e8) & need[t]
        out[t] = torch.where(g, b, out[t]); got[t] = g
    return out, need & ~got

def fill_rest(Z, miss, valid):
    """the still-missing character pixels: pull-push in disparity from the valid environment"""
    if not miss.any(): return Z
    disp = torch.where(valid & ~miss, 1.0 / Z.clamp_min(0.05), torch.zeros_like(Z)); w = (valid & ~miss).float()
    d2 = gblur(pushpull(disp, w), 3.0); return torch.where(miss, 1.0 / d2.clamp_min(1e-4), Z)

def shade(Z, valid, R, f, cx, cy, sig):
    """Lambert of the camera-facing normals (disparity blurred by sig px) under the fixed world light: [T,H,W] in 0..1"""
    T, H, W = Z.shape; dev = Z.device
    disp = gblur(torch.where(valid, 1.0 / Z.clamp_min(0.05), torch.full_like(Z, 1e-3)), sig); zs = 1.0 / disp.clamp_min(1e-4)
    v, u = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    P = torch.stack([((u - cx) / f)[None] * zs, -((v - cy) / f)[None] * zs, zs], -1)        # camera (right, up, forward)
    dx = torch.zeros_like(P); dy = torch.zeros_like(P); dx[:, :, 1:-1] = (P[:, :, 2:] - P[:, :, :-2]) / 2; dy[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2
    n = torch.cross(dx, dy, dim=-1); n = n / (n.norm(dim=-1, keepdim=True) + 1e-9); n = torch.where(((n * P).sum(-1, keepdim=True) > 0), -n, n)
    nw = torch.einsum("thwc,tcd->thwd", n, R); return torch.where(valid, (nw @ L.to(dev)).clamp(0, 1), torch.zeros_like(Z))

def compose(BASE, tier, sh):
    """BASE [T,H,W,3] uint8 (characters G >= 20), tier [T,H,W] u8, sh [T,H,W] float -> [T,H,W,3] uint8"""
    act = BASE[..., 1] >= 20; out = BASE.clone(); g = out[..., 1]
    g[tier == 1] = G_UNK; g[tier == 2] = G_COARSE; ex = tier == 3; g[ex] = torch.round(G_EX0 + (G_EX1 - G_EX0) * sh[ex]).to(torch.uint8)
    unk = tier == 1; out[..., 0][unk] = 0; out[..., 2][unk] = 0; out[tier == 0] = 0
    out[act] = BASE[act]
    halo = (F.max_pool2d(act[:, None].float(), 5, 1, 2)[:, 0] > 0) & ~act; out[halo] = 0
    return out

def splat(Pw, pos, R, f, cx, cy, H, W, grow=2, dense_iters=6):
    """(roam: no moderngl next to the skeleton renderer's EGL) world points [N,3] -> planar depth per frame [T,H,W] by a z-buffer (min),
    gaps of the point sampling closed by a min-filter of radius `grow` (only where something was hit nearby); 0 = nothing"""
    T = pos.shape[0]; out = torch.zeros(T, H, W, device=Pw.device)
    for t in range(T):
        rel = Pw - pos[t]; zc = rel @ R[t, 2]; k = zc > 0.05; rel, zc = rel[k], zc[k]
        u = torch.floor(cx + f * (rel @ R[t, 0]) / zc).long(); v = torch.floor(cy - f * (rel @ R[t, 1]) / zc).long()
        k = (u >= 0) & (u < W) & (v >= 0) & (v < H); buf = torch.full((H * W,), 1e9, device=Pw.device)
        buf.scatter_reduce_(0, v[k] * W + u[k], zc[k], reduce="amin"); b = buf.view(H, W)
        for _ in range(grow): b = torch.where(b < 1e8, b, -F.max_pool2d(-b[None, None], 3, 1, 1)[0, 0])
        for _ in range(dense_iters):                                                    # (10-18) magnified regions (camera moved closer) leave a dotted point cloud:
            hit = (b < 1e8).float()                                                     # close the gaps only where the local hit density says "a surface is here"
            dens = F.avg_pool2d(hit[None, None], 15, 1, 7)[0, 0]
            b = torch.where((b >= 1e8) & (dens > 0.15), -F.max_pool2d(-b[None, None], 3, 1, 1)[0, 0], b)
        out[t] = torch.where(b < 1e8, b, torch.zeros_like(b))
    return out

def splat_grid(Pg, ok, pos, R, f, cx, cy, H, W, rmax=32, ratio=1.5):
    """(10-18) the MoGe world as a SURFACE, not points: the per-pixel world grid Pg [h,w,3] (ok [h,w]) is projected into every frame and each
    point drawn as a square as large as the gap to its grid neighbours there (only across neighbours on the same surface: depth ratio < 1.12),
    z-buffered (min). Bucketed by radius, a min-filter per bucket. Returns planar depth [T,H,W], 0 = nothing."""
    T = pos.shape[0]; dev = Pg.device; out = torch.zeros(T, H, W, device=dev)
    zr = Pg.new_zeros(Pg.shape[:2])
    for t in range(T):
        rel = Pg - pos[t]; zc = rel @ R[t, 2]; u = cx + f * (rel @ R[t, 0]) / zc.clamp_min(1e-3); v = cy - f * (rel @ R[t, 1]) / zc.clamp_min(1e-3)
        good = ok & (zc > 0.05); gap = torch.zeros_like(zc)
        for du_, dv_ in ((0, 1), (1, 0), (1, 1)):                                       # right, down, diagonal neighbours on the same surface
            a = (slice(0, Pg.shape[0] - du_), slice(0, Pg.shape[1] - dv_)); b = (slice(du_, None), slice(dv_, None))
            same = good[a] & good[b] & (torch.maximum(zc[a], zc[b]) < ratio * torch.minimum(zc[a], zc[b]))   # grazing ground: adjacent rows differ a lot
            g = torch.where(same, torch.maximum((u[a] - u[b]).abs(), (v[a] - v[b]).abs()), torch.zeros_like(zc[a]))
            gap[a] = torch.maximum(gap[a], g)
        r = (gap / 2).ceil().clamp(0, rmax).long(); iu = torch.floor(u).long(); iv = torch.floor(v).long()
        inb = good & (iu >= -rmax) & (iu < W + rmax) & (iv >= -rmax) & (iv < H + rmax); best = torch.full((H, W), 1e9, device=dev)
        for rr in torch.unique(r[inb]).tolist():
            m = inb & (r == rr); pad = rr; buf = torch.full(((H + 2 * pad) * (W + 2 * pad),), 1e9, device=dev)
            uu = iu[m] + pad; vv = iv[m] + pad; k = (uu >= 0) & (uu < W + 2 * pad) & (vv >= 0) & (vv < H + 2 * pad)
            buf.scatter_reduce_(0, vv[k] * (W + 2 * pad) + uu[k], zc[m][k], reduce="amin"); b2 = buf.view(H + 2 * pad, W + 2 * pad)
            if rr > 0: b2 = -F.max_pool2d(-b2[None, None], 2 * rr + 1, 1, rr)[0, 0]
            best = torch.minimum(best, b2[pad:pad + H, pad:pad + W])
        out[t] = torch.where(best < 1e8, best, torch.zeros_like(best))
    return out
