"""Exp A main loop — counterfactual relight sensitivity vs. dense metric-depth error.

For each C3VD / SimCol frame we relight at FIXED geometry (dense_relight_families, depth unchanged)
and, per depth model, measure:
  (a) dE      = dense metric-depth error(relit) - error(original)   [per metric: absrel/rmse/silog/grad/d125]
  (b) S_relight = mean|zscore(pred_relit) - zscore(pred_orig)|       [how much the prediction moved]
The per-image affine align (a,b) is fit ONCE on the ORIGINAL prediction and FROZEN, then reused for every
relit prediction — so relit alignment cannot absorb (hide) an illumination-induced structural change.
Aggregate per family across frames with a SCENE-clustered bootstrap CI (resample sequence ids).

Interfaces consumed (built by sibling modules, imported lazily so this file import-checks standalone):
  expA_metrics      :: ss_fit(pred,gt,mask)->(a,b) ; apply(pred,a,b)->aligned ; metrics(aligned,gt,mask)->dict
  expA_dense_inference :: DensePredictor().predict(model_name, bgr_uint8)->pred_float HxW (native size)
  expA_controls     :: (control) predictor with .predict(model_name, bgr, gt=None) for {oracle,lumonly}
  dense_relight_families :: relight_dense(bgr,depth,family,param,seed,dataset)->relit_bgr ; realized_rho(...)

Run (GPU):
  python3 expA_run.py --model dav2 --dataset c3vd --families all --n_frames 120 \
      --out results/counterfactual/expA/expA_dav2_c3vd.json
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# --- project roots ---------------------------------------------------------
P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
C3VD_ROOT = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test"
SIMCOL_ROOT = "/well/rittscher/users/fxh757/Code/Depth_metrics_model/SyntheticColon_I"

ALL_FAMILIES = ["global", "spatial", "physics", "decorr", "invert"]
CONTROL_MODELS = {"oracle", "lumonly"}
# C3VD 16-bit raw depth (larger=farther) -> mm : raw/65535 * (65535*1.525e-6*1000) == raw * 1.525e-3, ~0..100mm
C3VD_MM_PER_RAW = 0.000001525 * 1000.0
DEPTH_RAW_MAX = 65535.0


# =========================================================================
# scene-clustered bootstrap CI  (pattern copied from e_phantom_decouple.py:74-86)
# =========================================================================
def seq_cluster_ci(vals, seqs, B=5000, seed=0):
    """Sequence-clustered bootstrap CI of the mean (resample sequences, average their frames)."""
    vals = np.asarray(vals, float)
    seqs = np.asarray(seqs)
    m = ~np.isnan(vals)
    vals, seqs = vals[m], seqs[m]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    uniq = np.unique(seqs)
    by = {s: vals[seqs == s] for s in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        boot[b] = np.concatenate([by[s] for s in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi), int(len(uniq))


def _ci(vals, seqs):
    m, lo, hi, nseq = seq_cluster_ci(vals, seqs)
    return {"mean": m, "lo": lo, "hi": hi, "nseq": nseq}


# =========================================================================
# dataset loaders  -> list of frame dicts {name, bgr, depth(far-large), gt(metric), mask(bool), seq}
# =========================================================================
def _c3vd_seq(fn):
    """C3VD sequence id = filename minus the trailing _NNNN frame index."""
    base = fn[:-4] if fn.endswith(".png") else fn
    return base.rsplit("_", 1)[0]


def _load_fov(mask_path, hw):
    """invalid_mask.png convention: white(255)=VALID. Resize (NEAREST) to native hw, return bool valid."""
    if not os.path.exists(mask_path):
        return np.ones(hw, bool)
    m = np.asarray(Image.open(mask_path).convert("L"))
    m = np.asarray(Image.fromarray(m).resize((hw[1], hw[0]), Image.NEAREST))
    return m > 127


def load_frames(dataset, n_frames):
    dataset = dataset.lower()
    if dataset == "c3vd":
        specs = _list_c3vd()
    elif dataset == "simcol":
        specs = _list_simcol()
    else:
        raise ValueError(f"unknown dataset {dataset!r} (want c3vd|simcol)")
    # spread selection across the sorted list so the scene clusters stay diverse
    if n_frames and n_frames > 0 and n_frames < len(specs):
        idx = np.linspace(0, len(specs) - 1, n_frames).astype(int)
        idx = np.unique(idx)
        specs = [specs[i] for i in idx]
    frames = []
    for sp in specs:
        fr = _decode(dataset, sp)
        if fr is not None:
            frames.append(fr)
    return frames


def _list_c3vd():
    img_dir = os.path.join(C3VD_ROOT, "images")
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    return [{"img": f,
             "dep": os.path.join(C3VD_ROOT, "depths", os.path.basename(f)),
             "name": os.path.basename(f),
             "seq": _c3vd_seq(os.path.basename(f))} for f in files]


def _list_simcol():
    out = []
    for seq_dir in sorted(glob.glob(os.path.join(SIMCOL_ROOT, "Frames_S*"))):
        seq = os.path.basename(seq_dir)
        for f in sorted(glob.glob(os.path.join(seq_dir, "images", "*.png"))):
            out.append({"img": f,
                        "dep": os.path.join(seq_dir, "depths", os.path.basename(f)),
                        "name": f"{seq}/{os.path.basename(f)}",
                        "seq": seq})
    return out


# cache FOV masks per (dataset, hw)
_FOV_CACHE = {}


def _decode(dataset, sp):
    if not os.path.exists(sp["dep"]):
        return None
    if dataset == "c3vd":
        bgr = cv2.imread(sp["img"])  # BGR uint8, native res
        if bgr is None:
            return None
        raw = np.asarray(Image.open(sp["dep"])).astype(np.float32)  # 16-bit, larger=farther
        gt = raw * C3VD_MM_PER_RAW  # mm, ~0..100
        hw = gt.shape
        key = ("c3vd", hw)
        if key not in _FOV_CACHE:
            _FOV_CACHE[key] = _load_fov(os.path.join(C3VD_ROOT, "invalid_mask.png"), hw)
        mask = _FOV_CACHE[key] & (gt > 0) & (gt <= 100.0)
        depth_far = gt  # larger=farther (same ordering) -> relight geometry
    else:  # simcol
        rgb = Image.open(sp["img"]).convert("RGB")  # native res
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        raw = np.asarray(Image.open(sp["dep"])).astype(np.float32)
        gt = raw / DEPTH_RAW_MAX  # normalized, larger=farther
        hw = gt.shape
        key = ("simcol", hw)
        if key not in _FOV_CACHE:
            _FOV_CACHE[key] = _load_fov(os.path.join(SIMCOL_ROOT, "invalid_mask.png"), hw)
        mask = _FOV_CACHE[key] & (gt > 0)
        depth_far = gt
    if mask.sum() < 500:
        return None
    return {"name": sp["name"], "seq": sp["seq"], "bgr": bgr,
            "depth": depth_far.astype(np.float32), "gt": gt.astype(np.float32), "mask": mask}


# =========================================================================
# predictor construction + robust call
# =========================================================================
def build_predictor(model_name):
    """Return a predictor object exposing .predict(model_name, bgr[, gt]). Controls -> expA_controls."""
    if model_name in CONTROL_MODELS:
        import expA_controls as CT
        for cand in ("ControlPredictor", "ControlsPredictor", "DensePredictor", "Predictor"):
            if hasattr(CT, cand):
                return getattr(CT, cand)()
        raise ImportError("expA_controls exposes no ControlPredictor/DensePredictor class")
    import expA_dense_inference as DI
    return DI.DensePredictor()


def _to_np(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x, dtype=np.float32)
    return np.squeeze(x)


_ACCEPTS_GT = {}


def _predict(predictor, model_name, bgr, gt=None):
    """Call predict, passing gt= only if the predictor's signature accepts it (oracle needs gt; the 7 real
    models take (name,bgr)). Signature is inspected once per predictor so an internal TypeError is NOT masked."""
    key = id(predictor)
    if key not in _ACCEPTS_GT:
        import inspect
        try:
            params = inspect.signature(predictor.predict).parameters
            _ACCEPTS_GT[key] = ("gt" in params) or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (ValueError, TypeError):
            _ACCEPTS_GT[key] = True  # builtins / C-callables: assume it tolerates gt
    p = predictor.predict(model_name, bgr, gt=gt) if _ACCEPTS_GT[key] else predictor.predict(model_name, bgr)
    return _to_np(p)


def _match(pred, hw):
    """Ensure the prediction matches the GT native HxW (safety net; DensePredictor should already return native)."""
    if pred.ndim != 2:
        pred = np.squeeze(pred)
    if pred.shape != tuple(hw):
        pred = cv2.resize(pred.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return pred.astype(np.float32)


# =========================================================================
# per-frame counterfactual measures
# =========================================================================
def _s_relight(pred, pred0, mask):
    """mean over mask of |zscore(pred) - zscore(pred0)| (per-image standardized -> scale/shift invariant)."""
    def z(x):
        v = x[mask].astype(np.float64)
        mu, sd = v.mean(), v.std()
        return (x.astype(np.float64) - mu) / (sd + 1e-8)
    d = np.abs(z(pred) - z(pred0))
    return float(d[mask].mean())


def realizations(family, n_seed):
    """(param,seed) grid per family. global/spatial vary by both kind(param%3) and seed; physics by seed;
    decorr/invert are deterministic given (bgr,depth) -> a single realization."""
    n_seed = max(1, int(n_seed))
    if family in ("global", "spatial"):
        return [(p, s) for p in (0, 1, 2) for s in range(n_seed)]
    if family == "physics":
        return [(0, s) for s in range(2 * n_seed)]
    return [(0, 0)]  # decorr, invert


# =========================================================================
# main
# =========================================================================
def main(model_name=None, dataset=None, families=None, n_frames=0, out_json=None,
         n_seed=2, argv=None):
    if model_name is None:
        ap = argparse.ArgumentParser(description="Exp A counterfactual relight main loop")
        ap.add_argument("--model", required=True,
                        help="dav2|endoomni|dac|unidepth|metric3d|zoedepth|ppsnet|oracle|lumonly")
        ap.add_argument("--dataset", required=True, choices=["c3vd", "simcol"])
        ap.add_argument("--families", default="all",
                        help="'all' or comma list of global,spatial,physics,decorr,invert")
        ap.add_argument("--n_frames", type=int, default=0, help="0 = all frames")
        ap.add_argument("--n_seed", type=int, default=2, help="stochastic realizations per family (see realizations())")
        ap.add_argument("--out", default=None, help="output JSON path (default results/counterfactual/expA/expA_<m>_<d>.json)")
        a = ap.parse_args(argv)
        model_name, dataset, n_frames, n_seed = a.model, a.dataset, a.n_frames, a.n_seed
        families = a.families
        out_json = a.out

    fam_list = ALL_FAMILIES if (families in (None, "all", "")) else [f.strip() for f in str(families).split(",") if f.strip()]
    for f in fam_list:
        if f not in ALL_FAMILIES:
            raise ValueError(f"unknown family {f!r} (want subset of {ALL_FAMILIES})")
    if out_json is None:
        out_json = os.path.join(P1, "results/counterfactual/expA", f"expA_{model_name}_{dataset}.json")
    if not os.path.isabs(out_json):
        out_json = os.path.join(P1, out_json)

    # lazy import of relight (sibling-independent, but keep top import clean for standalone import-check)
    import dense_relight_families as RLF

    frames = load_frames(dataset, n_frames)
    if not frames:
        raise RuntimeError(f"no frames loaded for dataset={dataset}")
    predictor = build_predictor(model_name)

    import expA_metrics as EM
    print(f"[expA] model={model_name} dataset={dataset} frames={len(frames)} families={fam_list} "
          f"n_seed={n_seed}", flush=True)

    # per-frame accumulators (one value per frame per family; NaN if that family failed on the frame)
    metric_keys = None
    acc = {fam: {"S": [], "rho": []} for fam in fam_list}          # dE_<k> added lazily once keys known
    seqs = []

    for fi, fr in enumerate(frames):
        bgr, depth, gt, mask = fr["bgr"], fr["depth"], fr["gt"], fr["mask"]
        hw = gt.shape
        # ---- original prediction -> FREEZE (a,b) ----
        try:
            pred0 = _match(_predict(predictor, model_name, bgr, gt=gt), hw)
        except Exception as e:
            print(f"  [skip frame {fr['name']}] predict(orig) failed: {type(e).__name__}: {e}", flush=True)
            continue
        maskf = mask & np.isfinite(pred0)          # crop models return NaN outside center FOV -> exclude + FREEZE
        if int(maskf.sum()) < 500:
            print(f"  [skip frame {fr['name']}] <500 finite-pred pixels", flush=True)
            continue
        a_fit, b_fit = EM.ss_fit(pred0, gt, maskf)
        err0 = EM.metrics(EM.apply(pred0, a_fit, b_fit), gt, maskf)
        if metric_keys is None:
            metric_keys = list(err0.keys())
            for fam in fam_list:
                for k in metric_keys:
                    acc[fam][f"dE_{k}"] = []
        seqs.append(fr["seq"])

        for fam in fam_list:
            dE_reals = {k: [] for k in metric_keys}
            S_reals, rho_reals = [], []
            for (param, seed) in realizations(fam, n_seed):
                try:
                    relit = RLF.relight_dense(bgr, depth, fam, param=param, seed=seed, dataset=dataset)
                    pred = _match(_predict(predictor, model_name, relit, gt=gt), hw)
                except Exception as e:
                    print(f"  [frame {fr['name']} {fam} p{param} s{seed}] failed: {type(e).__name__}: {e}", flush=True)
                    continue
                err = EM.metrics(EM.apply(pred, a_fit, b_fit), gt, maskf & np.isfinite(pred))
                for k in metric_keys:
                    dE_reals[k].append(err[k] - err0[k])
                S_reals.append(_s_relight(pred, pred0, maskf))
                try:
                    rho_reals.append(RLF.realized_rho(relit, depth, mask))
                except Exception:
                    rho_reals.append(float("nan"))
            # collapse realizations -> one value per frame per family
            for k in metric_keys:
                acc[fam][f"dE_{k}"].append(float(np.mean(dE_reals[k])) if dE_reals[k] else float("nan"))
            acc[fam]["S"].append(float(np.mean(S_reals)) if S_reals else float("nan"))
            acc[fam]["rho"].append(float(np.nanmean(rho_reals)) if rho_reals else float("nan"))

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frames)} frames", flush=True)

    if metric_keys is None:
        raise RuntimeError("no frame produced a valid original prediction")

    # ---- aggregate per family with scene-clustered bootstrap CI ----
    per_family = {}
    for fam in fam_list:
        entry = {}
        for k in metric_keys:
            entry[f"dE_{k}"] = _ci(acc[fam][f"dE_{k}"], seqs)
        entry["S_relight"] = _ci(acc[fam]["S"], seqs)
        rho_vals = np.asarray(acc[fam]["rho"], float)
        entry["rho_realized_mean"] = float(np.nanmean(rho_vals)) if np.isfinite(rho_vals).any() else float("nan")
        per_family[fam] = entry

    out = {
        "model": model_name,
        "dataset": dataset,
        "n_frames": len(seqs),
        "n_seed": n_seed,
        "families": fam_list,
        "metric_keys": metric_keys,
        "frozen_align": "ss(a,b) fit ONCE on original prediction, reused for all relit",
        "per_family": per_family,
        "seqs": seqs,
        "per_frame": {fam: {kk: acc[fam][kk] for kk in acc[fam]} for fam in fam_list},
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)

    # ---- console summary ----
    print(f"\n=== Exp A: {model_name} / {dataset}  ({len(seqs)} frames, "
          f"{per_family[fam_list[0]]['S_relight']['nseq']} seqs) ===")
    hdr = f"{'family':<9}{'dE_absrel[95%CI]':>26}{'dE_rmse[95%CI]':>26}{'S_relight[95%CI]':>26}{'rho':>8}"
    print(hdr)
    for fam in fam_list:
        e = per_family[fam]
        da = e.get("dE_absrel", {"mean": float('nan'), "lo": float('nan'), "hi": float('nan')})
        dr = e.get("dE_rmse", {"mean": float('nan'), "lo": float('nan'), "hi": float('nan')})
        s = e["S_relight"]
        print(f"{fam:<9}"
              f"{da['mean']:>10.4f}[{da['lo']:.3f},{da['hi']:.3f}]"
              f"{dr['mean']:>10.4f}[{dr['lo']:.3f},{dr['hi']:.3f}]"
              f"{s['mean']:>10.4f}[{s['lo']:.3f},{s['hi']:.3f}]"
              f"{e['rho_realized_mean']:>8.2f}")
    print(f"\nwrote {out_json}")
    return out


if __name__ == "__main__":
    main()
