"""Re-label a case's estimated (coarse, tier 2) geometry as exact (tier 3): the coarse-coded pixels of proxy.mkv get the
exact G channel (normal shading of the case's own world depth) and tier cells 2 -> 3. Same geometry, same cameras.

    python tools/case_to_exact.py <case dir> <new case dir>"""
import sys, os, shutil, subprocess, numpy as np, cv2
import wizard.data.cond_encode as E7
src, dst = sys.argv[1:3]; os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f not in ("proxy.mkv", "tier.npy"): shutil.copy(f"{src}/{f}", dst)
tier_all = np.load(f"{src}/tier.npy"); tier_all = np.where(tier_all == 2, 3, tier_all).astype(np.uint8); np.save(f"{dst}/tier.npy", tier_all)
w = np.load(f"{src}/world_px.npz"); Z = w["z"].astype(np.float32); g = np.load(f"{src}/gbuf.npz"); K = g["K"]; R = g["R"]
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{src}/proxy.mkv", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
V = np.frombuffer(raw, np.uint8).reshape(-1, 576, 1024, 3).copy(); n = min(len(V), len(Z)); f8 = float(K[0]) * 8; print("frames", n, "focal(832)", f8)
for t in range(n):
    z = Z[t]; ok = z > 0
    sh = E7.shade(np.where(ok, z, 0.0), f8, R[t].astype(float), ok)                        # [480,832] in the cropped 832 frame
    G = np.round(E7.G_EX0 + (E7.G_EX1 - E7.G_EX0) * sh).astype(np.uint8)
    G853 = np.zeros((480, 853), np.uint8); G853[:, 10:842] = G; ok853 = np.zeros((480, 853), bool); ok853[:, 10:842] = ok
    G1k = cv2.resize(G853, (1024, 576), interpolation=cv2.INTER_NEAREST); ok1k = cv2.resize(ok853.astype(np.uint8), (1024, 576), interpolation=cv2.INTER_NEAREST) > 0
    m = ok1k & (np.abs(V[t][..., 1].astype(int) - E7.G_COARSE) <= 2)
    V[t][..., 1][m] = G1k[m]
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1024x576", "-r", "30", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0", f"{dst}/proxy.mkv"], input=V.tobytes(), check=True)
print("EXACT_CASE", dst)
