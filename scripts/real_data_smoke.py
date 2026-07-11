#!/usr/bin/env python
"""
Real-data full-pass smoke test for MetabTravLR development.

Runs the ACTUAL SpaceShip pipeline end-to-end on a small real demo dataset with
gene focusing on, so we can eyeball that everything integrates: preprocessing +
CellOracle GRN (restricted to focus genes) + NicheNet + per-gene training +
betadata output + reading the coefficients back.

This is a manual/dev smoke script, NOT part of the pytest suite (run it by hand;
it does real training and may need network for the NicheNet download).

Usage:
    ~/miniconda3/envs/spacetravlr_env/bin/python scripts/real_data_smoke.py \
        [--data data/snrna_germinal_center.h5ad] [--genes CD74 BCL6 FOXO1] \
        [--epochs 5] [--outdir <tmp>]

Exit code 0 = the focus genes trained and produced readable betadata.
"""
import argparse, os, sys, time, glob, tempfile, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import scanpy as sc  # noqa: E402
import pandas as pd  # noqa: E402


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/snrna_germinal_center.h5ad")
    ap.add_argument("--genes", nargs="+", default=["CD74", "BCL6", "FOXO1"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--annot", default="cell_type")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--run-commot", action="store_true")
    args = ap.parse_args()

    from SpaceTravLR.spaceship import SpaceShip

    outdir = args.outdir or tempfile.mkdtemp(prefix="spacetravlr_smoke_")
    banner(f"Real-data smoke | data={args.data} | focus genes={args.genes} | "
           f"epochs={args.epochs} | outdir={outdir}")

    adata = sc.read_h5ad(args.data)
    print(f"loaded adata {adata.shape} | obs cols: {list(adata.obs.columns)} | "
          f"layers: {list(adata.layers.keys())}")
    missing = [g for g in args.genes if g not in adata.var_names]
    if missing:
        print(f"WARNING: focus genes not in var_names (will be dropped): {missing}")
    present = [g for g in args.genes if g in adata.var_names]
    assert present, "none of the requested focus genes are in the dataset"

    ship = SpaceShip(name="smoke", outdir=outdir, genes=present)
    assert ship.focus_genes == present, "focus_genes not stored"

    # --- full setup (preprocess -> CellOracle[focus-restricted] -> NicheNet) ---
    banner("STAGE 1/3: SpaceShip.setup_() (preprocess + CellOracle + NicheNet)")
    t0 = time.time()
    try:
        ship.setup_(adata, overwrite=True, run_commot=args.run_commot)
        print(f"setup_ OK in {time.time()-t0:.1f}s")
    except Exception:
        print("setup_ FAILED:\n" + traceback.format_exc())
        print("NOTE: if this is a NicheNet download error, re-run with network, "
              "or point tflinks at a cached parquet.")
        return 2

    # sanity on the CellOracle restriction: links should only cover focus targets
    try:
        n_link_targets = {t for df in ship.links.values() for t in df["target"].unique()}
        extra = n_link_targets - set(present)
        print(f"CellOracle links target genes = {sorted(n_link_targets)} "
              f"(focus={present}); non-focus targets present: {sorted(extra) or 'none'}")
    except Exception:
        print("could not introspect ship.links:\n" + traceback.format_exc())

    # --- training (only the focus genes) ---
    banner("STAGE 2/3: run_spacetravlr() -- trains ONLY the focus genes")
    t0 = time.time()
    try:
        ship.run_spacetravlr(max_epochs=args.epochs)
        print(f"run_spacetravlr OK in {time.time()-t0:.1f}s")
    except Exception:
        print("run_spacetravlr FAILED:\n" + traceback.format_exc())
        return 3

    beta_dir = f"{outdir}/betadata"
    produced = sorted(os.path.basename(p) for p in glob.glob(f"{beta_dir}/*_betadata.parquet"))
    orphans = sorted(os.path.basename(p) for p in glob.glob(f"{beta_dir}/*.orphan"))
    print(f"betadata files: {produced}")
    print(f"orphan files:   {orphans}")

    trained_genes = {p.split('_betadata')[0] for p in produced}
    orphan_genes = {o.rsplit('.orphan', 1)[0] for o in orphans}
    considered = trained_genes | orphan_genes
    unexpected = considered - set(present)
    assert not unexpected, f"non-focus genes were trained/attempted: {unexpected}"
    print(f"OK: only focus genes considered ({sorted(considered)}); "
          f"trained={sorted(trained_genes)}, orphaned={sorted(orphan_genes)}")

    # --- read coefficients back (the actual downstream use) ---
    banner("STAGE 3/3: read a betadata parquet (coefficient analysis surface)")
    if produced:
        one = f"{beta_dir}/{produced[0]}"
        bd = pd.read_parquet(one)
        beta_cols = [c for c in bd.columns if c.startswith("beta")]
        print(f"{produced[0]}: shape={bd.shape}, {len(beta_cols)} beta columns")
        print(f"  sample beta cols: {beta_cols[:8]}")
        print(f"  finite betas: {int(bd[beta_cols].notna().values.sum())} / "
              f"{bd[beta_cols].size}; nonzero: {int((bd[beta_cols].fillna(0).values != 0).sum())}")
    else:
        print("no betadata produced (all focus genes orphaned) -- check GRN coverage")

    banner("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
