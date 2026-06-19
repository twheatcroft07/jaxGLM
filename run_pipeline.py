"""jaxGLM command-line pipeline: config.yaml -> fitted GLM + kernels + significance, saved to a
project directory. The turnkey path over the library (design / poisson_glm / encoding / viz).

Usage:
    python run_pipeline.py --new NAME [PARENT_DIR]     # scaffold a project (then edit config.yaml)
    python run_pipeline.py PATH/TO/config.yaml         # run the pipeline

Run the fit on a GPU node (the solver needs jax[cuda]); scaffolding/--new is fine on a login node.
The config schema and data contract are documented in project.py.
"""
import os
import sys
import glob
import argparse
import numpy as np

import project as proj


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_data(cfg, base):
    """Concatenate the project's CSVs and return (df, session_groups, trial_ids|None). Rows keep
    file/temporal order within each session so shift-kernels are built on a contiguous grid."""
    import pandas as pd
    d = cfg["data"]
    files = sorted(glob.glob(_resolve(d["glob"], base)))
    if not files:
        raise FileNotFoundError(f"no CSVs matched {d['glob']!r} under {base}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if d.get("time_col"):                                  # stable sort within each session
        df = df.sort_values([d["session_col"], d["time_col"]], kind="stable").reset_index(drop=True)
    session = df[d["session_col"]].to_numpy()
    label_cols = [session]
    trial = None
    if d.get("trial_col") and d["trial_col"] in df.columns:
        trial = df[d["trial_col"]].to_numpy()
        label_cols.append(trial)
    from design import group_ids_from_labels
    # boundary-aware shifting bounds on SESSION only (kernels may span trials, never sessions)
    session_groups = group_ids_from_labels(session)
    trial_groups = group_ids_from_labels(*label_cols) if trial is not None else None
    print(f"  loaded {len(files)} file(s), {len(df)} bins, "
          f"{len(np.unique(session_groups))} session(s)"
          + (f", {len(np.unique(trial_groups))} trials" if trial_groups is not None else ""))
    return df, session_groups, trial_groups


def build_design_from_cfg(cfg, df, session_groups):
    import design as dz
    g = cfg["glm"]
    preds = cfg["data"]["predictors"]
    missing = [p for p in preds if p not in df.columns]
    if missing:
        raise ValueError(f"predictors not in data: {missing}")
    predictors = {p: df[p].to_numpy(dtype=float) for p in preds}
    lo, hi = g["shift_default"]
    bounds = g.get("shift_bounds") or {}
    shifts = {p: list(range(*(lambda b: (b[0], b[1] + 1))(bounds.get(p, [lo, hi])))) for p in preds}
    X, names = dz.build_design(predictors, shifts, groups=session_groups)   # boundary-aware
    return X, names


def select_responses(cfg, df):
    d = cfg["data"]
    if d.get("responses"):
        cols = d["responses"]
    else:
        cols = [c for c in df.columns if c.startswith(d["responses_prefix"])]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"response columns not in data: {missing}")
    if not cols:
        raise ValueError("no response columns selected")
    return cols, df[cols].to_numpy(dtype=float)


def trial_fold_ids(trial_groups, n_folds):
    """Assign whole trials to folds (contiguous blocks of trials) -> per-bin fold id, or None."""
    if trial_groups is None:
        return None
    uniq = np.unique(trial_groups)
    trial_to_fold = {t: int(i * n_folds / len(uniq)) for i, t in enumerate(uniq)}
    return np.array([trial_to_fold[t] for t in trial_groups])


def run(config_path):
    import jax.numpy as jnp
    import poisson_glm as pg
    import encoding as enc
    import design as dz

    base = os.path.dirname(os.path.abspath(config_path))
    cfg = proj.load_config(config_path)
    g, cv, sig, out = cfg["glm"], cfg["cv"], cfg["significance"], cfg["outputs"]
    family, l1 = g["family"], float(g["l1_ratio"])
    results_dir = _resolve("results", cfg["project"].get("path", base))
    models_dir = _resolve("models", cfg["project"].get("path", base))
    os.makedirs(results_dir, exist_ok=True); os.makedirs(models_dir, exist_ok=True)

    print("[1/6] loading data"); df, session_groups, trial_groups = load_data(cfg, base)
    print("[2/6] building design (boundary-aware shifts)")
    X, names = build_design_from_cfg(cfg, df, session_groups)
    resp_names, Y = select_responses(cfg, df)

    keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)   # drop bins with any NaN (post-shift)
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} bin(s) with NaN; {int(keep.sum())} remain")
    X, Y = X[keep], Y[keep]
    tg = trial_groups[keep] if trial_groups is not None else None
    print(f"  X {X.shape} ({len(names)} kernels), Y {Y.shape} ({len(resp_names)} responses)")

    kw = dict(L0=1.0, max_iter=int(g["max_iter"]), tol=float(g["tol"]))
    fold_ids = trial_fold_ids(tg, int(cv["n_folds"])) if cv.get("group_by_trial") else None

    import validate                                         # fail loud on a degenerate design
    validate.check_design(X, Y, family=family, names=names, fold_ids=fold_ids)

    Xz, mean, std = pg.standardize(jnp.asarray(X)) if g.get("standardize", True) else (
        jnp.asarray(X), None, None)
    Yj = jnp.asarray(Y)

    print("[3/6] selecting regularization (alpha)")
    alpha_spec = g["alpha"]
    if alpha_spec == "cv" or isinstance(alpha_spec, list):
        alphas = (enc.alpha_grid(Xz, Yj, l1_ratio=l1) if alpha_spec == "cv"
                  else np.asarray(alpha_spec, dtype=float))
        alpha_star, _ = enc.cv_select_alpha(Xz, Yj, alphas, l1, fold_ids, int(cv["n_folds"]),
                                            family, **kw)
        print(f"  alpha per unit: median {np.median(alpha_star):.4g} "
              f"[{alpha_star.min():.3g}, {alpha_star.max():.3g}]")
        W, b, n_iter, conv = pg.fit_units_alpha(Xz, Yj, jnp.asarray(alpha_star), l1,
                                                *kw.values(), family)
    else:
        alpha_star = np.full(Y.shape[1], float(alpha_spec))
        W, b, n_iter, conv = pg.fit_units(Xz, Yj, float(alpha_spec), l1, *kw.values(), family)
        print(f"  fixed alpha {float(alpha_spec):.4g}")

    health = validate.fit_health(W, b, conv, n_iter, int(g["max_iter"]), names=resp_names)
    print(f"  fit health: {health['n_converged']}/{health['n_units']} converged "
          f"(median {health['median_iters']} iters)"
          + (f", {health['n_extreme']} with extreme weights" if health['n_extreme'] else ""))

    mu = pg.predict_rate(Xz, W, b, family)
    mu0 = jnp.broadcast_to(Yj.mean(0)[None, :], Yj.shape)
    d2, _, _ = pg.frac_deviance_explained(Yj, mu, mu0, family)   # returns (d2, dev_m, dev_0)
    d2 = np.asarray(d2)
    print(f"  in-sample D2: median {np.median(d2):.3f}")

    # un-standardize weights so saved kernels are in raw-X units
    if mean is not None:
        W_raw, b_raw = pg.unstandardize_weights(W, b, mean, std)
    else:
        W_raw, b_raw = W, b

    print("[4/6] predictor importance" if out.get("predictor_importance") else "[4/6] (skip importance)")
    importance = {}
    if out.get("predictor_importance"):
        subsets = {p: [i for i, nm in enumerate(names) if nm.rsplit("_", 1)[0] == p]
                   for p in cfg["data"]["predictors"]}
        alphas = enc.alpha_grid(Xz, Yj, l1_ratio=l1)
        imp = enc.predictor_importance(Xz, Yj, subsets, alphas, l1, fold_ids,
                                       int(cv["n_folds"]), family, **kw)
        importance = {p: np.asarray(v) for p, v in imp["delta_d2"].items()}
        for p, v in importance.items():
            print(f"    {p:20s} median Delta-D2 = {np.median(v):+.4f}")

    print("[5/6] significance")
    wilcoxon = permutation = None
    alphas = enc.alpha_grid(Xz, Yj, l1_ratio=l1)
    if sig.get("run_wilcoxon", True):
        wilcoxon = enc.wilcoxon_full_vs_null(Xz, Yj, alphas, l1, fold_ids,
                                             int(cv["n_folds"]), family, **kw)
        print(f"    wilcoxon: {int(np.sum(wilcoxon['significant']))}/{Y.shape[1]} units p<0.05")
    if sig.get("run_permutation", False):
        # fast path: reuse the per-unit CV alpha as a FIXED regularization for observed+shuffles
        permutation = enc.permutation_null_d2(Xz, Yj, alpha_star, l1, fold_ids, int(cv["n_folds"]),
                                              family, n_perm=int(sig["n_perm"]), **kw)
        print(f"    permutation: {int(np.sum(permutation['significant']))}/{Y.shape[1]} units p<0.05")

    print("[6/6] saving outputs")
    fit_path = os.path.join(models_dir, "fit.npz")
    save = dict(W=np.asarray(W_raw), b=np.asarray(b_raw), names=np.array(names),
                responses=np.array(resp_names), alpha=np.asarray(alpha_star), l1_ratio=l1,
                family=family, d2=d2, converged=np.asarray(conv))
    for p, v in importance.items():
        save[f"delta_d2__{p}"] = v
    if wilcoxon is not None:
        save["wilcoxon_p"] = np.asarray(wilcoxon["pvalue"])
    if permutation is not None:
        save["permutation_p"] = np.asarray(permutation["pvalue"])
    np.savez(fit_path, **save)
    print(f"    -> {fit_path}")

    # per-response summary CSV
    import csv
    with open(os.path.join(results_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); head = ["response", "alpha", "D2"]
        if wilcoxon is not None: head.append("wilcoxon_p")
        if permutation is not None: head.append("permutation_p")
        w.writerow(head)
        for i, r in enumerate(resp_names):
            row = [r, f"{alpha_star[i]:.6g}", f"{d2[i]:.4f}"]
            if wilcoxon is not None: row.append(f"{wilcoxon['pvalue'][i]:.4g}")
            if permutation is not None: row.append(f"{permutation['pvalue'][i]:.4g}")
            w.writerow(row)
    print(f"    -> {os.path.join(results_dir, 'summary.csv')}")

    if out.get("kernels_plot", True):
        import viz
        bw = None
        fig = viz.plot_kernels(np.asarray(W_raw), names, bin_width=bw, mean_sem=Y.shape[1] > 1)
        fig.savefig(os.path.join(results_dir, "kernels.png"), dpi=110, bbox_inches="tight")
        print(f"    -> {os.path.join(results_dir, 'kernels.png')}")
    if out.get("reconstruction_plot", True):
        import viz
        fig = viz.plot_reconstruction(np.asarray(Y), np.asarray(mu), unit=0, bin_width=None)
        fig.savefig(os.path.join(results_dir, "reconstruction.png"), dpi=110, bbox_inches="tight")
        print(f"    -> {os.path.join(results_dir, 'reconstruction.png')}")
    print("done.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="jaxGLM config-driven pipeline")
    ap.add_argument("config", nargs="?", help="path to config.yaml")
    ap.add_argument("--new", metavar="NAME", help="scaffold a new project and exit")
    ap.add_argument("parent", nargs="?", default=".", help="parent dir for --new (default: .)")
    args = ap.parse_args(argv)
    if args.new:
        path = proj.create_new_project(args.new, args.config or args.parent)
        print(f"created project at {path}\n  1. edit {os.path.join(path, 'config.yaml')}"
              f"\n  2. put CSVs in {os.path.join(path, 'data')}/"
              f"\n  3. run: python run_pipeline.py {os.path.join(path, 'config.yaml')}")
        return
    if not args.config:
        ap.error("provide a config.yaml, or use --new NAME to scaffold one")
    run(args.config)


if __name__ == "__main__":
    main()
