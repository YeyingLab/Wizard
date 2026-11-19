"""Proxy-video encoder (numpy reference; shared by every data source, and inference calls the same shade()).
A proxy frame (1024x576 RGB) keeps the base condition's R (height) / B (depth) and characters, with
  G = tier:  sky 0 | unknown: R=B=0, G=38 | coarse: G=77 | exact: G = 153..255 by the normal shading (fixed world light)
  + a 2 px black outline round the characters.
Normals: camera-space from the planar depth (disparity smoothed sigma = 5 px at 1280 wide), to world with the camera rows (right, up, forward; y up)."""
import numpy as np, cv2
L = np.array([0.45, 0.84, -0.30]); L /= np.linalg.norm(L)                        # world y up
G_UNK, G_COARSE, G_EX0, G_EX1 = 38, 77, 153, 255

def shade(z, f, Rrows, valid, sig1280=5.0):
    """z [h,w] planar depth (m), f focal (px at this resolution), Rrows [3,3] right/up/forward (world, y up) -> Lambert [h,w] in 0..1"""
    h, w = z.shape; sig = sig1280 * w / 1280.0
    disp = cv2.GaussianBlur((1.0 / np.maximum(np.where(valid, z, 1e3), 0.05)).astype(np.float32), (0, 0), sig); zs = 1.0 / np.maximum(disp, 1e-4)
    u = (np.arange(w) + 0.5 - w / 2) / f; v = (np.arange(h) + 0.5 - h / 2) / f
    P = np.stack([u[None, :] * zs, -v[:, None] * zs, zs], -1).astype(np.float32)        # camera (right, up, forward)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0)); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = np.where(((n * P).sum(-1) > 0)[..., None], -n, n)                             # facing the camera
    nw = n @ np.asarray(Rrows, float)                                                   # world
    return np.where(valid, np.clip(nw @ L, 0, 1), 0.0)

def fill_under(z, mask, valid):
    """depth under the characters from the surrounding environment (nearest + blur, in disparity)"""
    from scipy import ndimage as ndi
    m = mask & valid
    if not m.any(): return z
    disp = np.where(valid & ~m, 1.0 / np.maximum(z, 0.05), 0.0); src = valid & ~m
    _, (iy, ix) = ndi.distance_transform_edt(~src, return_indices=True); d2 = disp[iy, ix]
    d2 = np.where(m, cv2.GaussianBlur(d2.astype(np.float32), (0, 0), 6.0), d2); return np.where(m, 1.0 / np.maximum(d2, 1e-4), z)

SHIELD_PART = 19
def draw_shield(base, disc):
    """(user: 把盾画出来) the player's shield: a filled disc in the player's colour, G = slot 1 * 32 + SHIELD_PART; disc = (u, v, r) in the
    832x480 crop (windows/shield) -> 1024x576 (crop x0 = 10.5 of 853 wide, scale 1.2). Drawn into the base frame BEFORE compose, so it is a character pixel."""
    u_, w_, r_ = disc
    if not np.isfinite(u_): return base
    v = base.copy(); pl = (v[..., 1] >= 34) & (v[..., 1] < 148); col = np.median(v[pl], 0) if pl.any() else np.array([20, 0, 230])
    cv2.circle(v, (int(round((u_ + 10.5) * 1.2)), int(round(w_ * 1.2))), max(2, int(round(r_ * 1.2))), (int(col[0]), 32 + SHIELD_PART, int(col[2])), -1); return v

def compose(base, tier, sh=None):
    """base [576,1024,3] uint8; tier [576,1024] uint8 0 sky 1 unknown 2 coarse 3 exact; sh [576,1024] shading (exact)"""
    act = base[..., 1] >= 20; out = base.copy(); g = out[..., 1]
    g[tier == 1] = G_UNK; g[tier == 2] = G_COARSE
    if sh is not None: ex = tier == 3; g[ex] = np.round(G_EX0 + (G_EX1 - G_EX0) * sh[ex]).astype(np.uint8)
    out[tier == 1, 0] = 0; out[tier == 1, 2] = 0; out[tier == 0] = 0
    out[act] = base[act]                                                                  # characters on top
    halo = cv2.dilate(act.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~act; out[halo] = 0
    return out

def temporal_fill(cam_t, z_t, mask_t, srcs, step=2):
    """(10-17, user: 修) the environment behind the characters from OTHER frames: every source frame's background points (its own characters excluded),
    reprojected into frame t, z-buffered; mask pixels they reach get that depth, the rest stays for fill_under.
    cam = (pos [3], Rrows [3,3] right/up/forward, f, cx, cy) in one world (y up, metres); z planar depth; srcs = [(cam, z, bg_valid)]"""
    h, w = z_t.shape; buf = np.full(h * w, np.inf)
    pos_t, R_t, f_t, cx_t, cy_t = cam_t
    for (pos, R, f, cx, cy), z, ok in srcs:
        vv, uu = np.mgrid[0:z.shape[0]:step, 0:z.shape[1]:step]; zz = z[::step, ::step]; m = ok[::step, ::step]
        u = uu[m] + 0.5; v = vv[m] + 0.5; zz = zz[m]
        P = pos + ((u - cx) / f * zz)[:, None] * R[0] - ((v - cy) / f * zz)[:, None] * R[1] + zz[:, None] * R[2]
        rel = P - pos_t; zc = rel @ R_t[2]; keep = zc > 0.1
        ut = cx_t + f_t * (rel[keep] @ R_t[0]) / zc[keep]; vt = cy_t - f_t * (rel[keep] @ R_t[1]) / zc[keep]; zc = zc[keep]
        iu = np.floor(ut).astype(int); iv = np.floor(vt).astype(int); inb = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
        iu, iv, zc = iu[inb], iv[inb], zc[inb]; sel = mask_t[iv, iu]
        np.minimum.at(buf, iv[sel] * w + iu[sel], zc[sel])
    buf = buf.reshape(h, w); got = np.isfinite(buf)
    # close the splat gaps (points every `step` px): min-filter the reached depth over a small window, only inside the mask
    k = 2 * step + 1; b2 = np.where(got, buf, 1e9).astype(np.float32); b2 = -cv2.dilate(-b2, np.ones((k, k), np.uint8)); got2 = (b2 < 1e8) & mask_t
    zf = np.where(got2, b2, z_t); return zf, got2

def draw_shield_853(base, disc, x0=10):
    """the shield disc (u, v, r in the 832x480 crop) into a 853x480 frame"""
    u_, w_, r_ = disc
    if not np.isfinite(u_): return base
    v = base.copy(); pl = (v[..., 1] >= 34) & (v[..., 1] < 148); col = np.median(v[pl], 0) if pl.any() else np.array([20, 0, 230])
    cv2.circle(v, (int(round(u_ + x0)), int(round(w_))), max(2, int(round(r_))), (int(col[0]), 32 + SHIELD_PART, int(col[2])), -1); return v
