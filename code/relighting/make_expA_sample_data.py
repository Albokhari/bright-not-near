"""Render qualitative Exp A samples: for a few frames and models, save the relit RGB and the model's
ss-aligned depth prediction under each relight family (shared colormap per frame so the reader SEES the
prediction change). GPU. Output: results/counterfactual/expA_samples/<model>/f<idx>_<family>_{rgb,dep}.png
+ f<idx>_gt.png / f<idx>_orig_rgb.png. A CPU compositor (make_expA_samples_fig.py) tiles these."""
import os, sys, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import expA_run as ER
import expA_metrics as EM
import dense_relight_families as RLF

P1 = ER.P1 if hasattr(ER, "P1") else "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
OUT = os.path.join(P1, "results/counterfactual/expA_samples")
FAMS = ["global", "spatial", "physics", "decorr", "invert"]
# representative C3VD frame indices (spread across sequences; stable ordering from load_frames)
DEFAULT_IDX = [8, 40, 90, 150]


def turbo(depth, mask, lo, hi):
    """Colorize a depth map (larger=farther) with turbo (near=warm, far=cool), fixed [lo,hi] scale."""
    d = np.clip((depth - lo) / max(1e-6, hi - lo), 0, 1)
    d = 1.0 - d                                    # invert so NEAR (small depth) = warm end
    rgb = (cm.turbo(d)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = (245, 245, 245)                   # grey out invalid/FOV
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dav2")
    ap.add_argument("--dataset", default="c3vd")
    ap.add_argument("--idx", default=",".join(map(str, DEFAULT_IDX)))
    ap.add_argument("--size", type=int, default=240, help="saved panel size (px)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    want = [int(i) for i in args.idx.split(",")]

    # decode ONLY the requested frames (load_frames(0) would eagerly decode all ~2889 -> OOM)
    specs = ER._list_c3vd() if args.dataset.lower() == "c3vd" else ER._list_simcol()
    frames = [ER._decode(args.dataset, specs[i]) for i in want if i < len(specs)]
    frames = [f for f in frames if f is not None]
    print(f"[samples] {len(frames)} frames x {len(models)} models x {len(FAMS)} families", flush=True)

    def rs(img):
        return cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)

    def csq(img):
        """Central min(H,W)-square crop. Crop models (Resize(shorter->518)+CenterCrop-518) predict ONLY this
        central square, so their native-canvas map is NaN in the side margins; C3VD is 1080x1350 (1.25 aspect).
        Cropping RGB/GT/depth all to this same square gives aligned, square panels with no side-NaN strips and
        no aspect squish (square->square) -- fixes the 'compressed and cropped from the sides' depth panels."""
        H, W = img.shape[:2]
        S = min(H, W); t = (H - S) // 2; l = (W - S) // 2
        return img[t:t + S, l:l + S]

    for mi, model in enumerate(models):
        od = os.path.join(OUT, model); os.makedirs(od, exist_ok=True)
        predictor = ER.build_predictor(model)
        for fi, fr in enumerate(frames):
            bgr, depth, gt, mask = fr["bgr"], fr["depth"], fr["gt"], fr["mask"]
            gtc, maskc = csq(gt), csq(mask)
            lo, hi = np.percentile(gtc[maskc], [2, 98])   # colour scale over the displayed (central-square) region
            # save GT + original RGB once (per frame, model-independent visuals reused)
            if mi == 0:
                cv2.imwrite(os.path.join(OUT, f"f{fi}_gt.png"), rs(turbo(gtc, maskc, lo, hi)))
                cv2.imwrite(os.path.join(OUT, f"f{fi}_orig_rgb.png"), rs(csq(bgr)))
            try:
                pred0 = ER._match(ER._predict(predictor, model, bgr, gt=gt), gt.shape)
            except Exception as e:
                print(f"  skip {model} f{fi}: {type(e).__name__}: {e}", flush=True); continue
            maskf = mask & np.isfinite(pred0)                       # alignment uses the full native valid region
            a, b = EM.ss_fit(pred0, gt, maskf)
            cv2.imwrite(os.path.join(od, f"f{fi}_original_dep.png"),
                        rs(turbo(csq(EM.apply(pred0, a, b)), csq(maskf), lo, hi)))
            for fam in FAMS:
                relit = RLF.relight_dense(bgr, depth, fam, param=0, seed=1, dataset=args.dataset)
                try:
                    pred = ER._match(ER._predict(predictor, model, relit, gt=gt), gt.shape)
                except Exception as e:
                    print(f"  skip {model} f{fi} {fam}: {e}", flush=True); continue
                mf = maskf & np.isfinite(pred)
                cv2.imwrite(os.path.join(od, f"f{fi}_{fam}_dep.png"),
                            rs(turbo(csq(EM.apply(pred, a, b)), csq(mf), lo, hi)))
                if mi == 0:                          # relit RGB is model-independent -> save once
                    cv2.imwrite(os.path.join(OUT, f"f{fi}_{fam}_rgb.png"), rs(csq(relit)))
            print(f"  done {model} f{fi}", flush=True)
    print(f"[samples] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
