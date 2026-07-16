"""Summarize harreman (metabolite crosstalk) outputs into human-readable spreadsheets.

Given the path to one ``easy_download/harreman_outputs`` folder produced by
``harreman_funcs.HarremanRunner``, this builds two tables:

* **master metabolite sheet** (one row per metabolite) — is the metabolite exchanged
  at all (cell-type-independent), and, for each cell-type-annotation *tier*, which
  cell-type interactions are significant. Because every tier here is a
  ``<T-cell subtype> vs "other"`` annotation, the tier columns are framed around
  T cells: ``within_Tcell`` (exchanged *within* a single T-cell type) and
  ``Tcell_interfaces`` (exchanged at the interface *between* a T-cell type and another
  type). Columns per tier are fixed regardless of how many labels the tier has, so
  Tier3 (5 labels) does not explode the sheet.
* **gene-pair long sheet** (one row per significant ``tier x cell-type-pair x gene
  pair``) — the drill-down of which transporter genes are co-expressed at each
  T-cell interface, plus the metabolite(s) each pair supports.

Directionality note (IMPORTANT): the ``Cell Type 1 / Cell Type 2`` ordering is **NOT**
a metabolite flux direction. harreman's ``compute_gene_pairs`` builds cell-type pairs
with ``itertools.combinations_with_replacement`` (when ``fix_ct`` is unset), so each
*unordered* pair is computed once, in **sorted label order** — the reverse ordering is
never tested. Combined with every transporter gene being typed ``IMP-EXP``
(bidirectional) and both gene orders being folded into each pair, the score is an
**undirected** spatial co-expression of a metabolite's transporters across the
CT1-CT2 neighbor interface (or within a type, on the diagonal). We therefore report
interactions as undirected ``A--B`` interfaces / ``A (self)`` diagonals, never in/out.
See ``DataForClaude/documentation/05_harreman_reference.md``.

Significance everywhere uses the ``selected`` flag, which the runner sets from the
**non-parametric FDR** (``FDR_np``); we also carry ``FDR_np`` through for context.

Scope: this operates on a *single* folder. A later multi-sample manager can call
``summarize_harreman_folder(..., sample_id=...)`` per sample and ``pd.concat`` the results.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---- filenames written by HarremanRunner.save_harreman_outputs -----------------------
NETWORK_JSON = "harreman_network.json"
CELL_INDEP_M = "[ccc_results][cell_com_df_m].csv"          # metabolite level, no cell types
CELL_INDEP_GP = "[ccc_results][cell_com_df_gp_sig].csv"    # gene-pair level, no cell types
CT_M = "[ct_ccc_results][cell_com_df_m].csv"               # per-tier, metabolite level
CT_GP = "[ct_ccc_results][cell_com_df_gp_sig].csv"         # per-tier, gene-pair level

DASH = "–"  # en-dash for undirected interface strings (A–B)


# ======================================================================================
# small helpers
# ======================================================================================
def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a 'selected'-style column (bool / 'True' / 1) to a clean boolean Series."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _fmt_fdr(x: float) -> str:
    try:
        return f"{float(x):.1e}"
    except (TypeError, ValueError):
        return "na"


def _pair_to_metabolites(network: dict) -> dict[frozenset, list[str]]:
    """Map each transporter gene pair (order-agnostic) -> list of metabolites it supports."""
    pair2metab: dict[frozenset, list[str]] = defaultdict(list)
    for metab, info in network.get("gp_per_metabolite", {}).items():
        for g1, g2 in info.get("gene_pair", []):
            pair2metab[frozenset((g1, g2))].append(metab)
    return pair2metab


def _is_tcell(label: str, background_label: str) -> bool:
    """In this T-cell-centric annotation scheme, any label that is not the background
    (``"other"``) is a T-cell (sub)type."""
    return isinstance(label, str) and label.strip().lower() != background_label.strip().lower()


def _discover_tiers(root: Path) -> list[str]:
    """Tier subfolders are those containing a per-tier metabolite CSV. Sorted by name."""
    tiers = [p.name for p in root.iterdir() if p.is_dir() and (p / CT_M).is_file()]
    return sorted(tiers)


# ======================================================================================
# core
# ======================================================================================
def summarize_harreman_folder(
    easy_download_path: str | Path,
    background_label: str = "other",
    sample_id: str | None = None,
):
    """Build (master_df, genepair_df) from one harreman_outputs folder.

    Parameters
    ----------
    easy_download_path : path to the ``harreman_outputs`` folder (the one containing
        ``harreman_network.json`` and the ``[ccc_results]...`` CSVs). A path to the
        parent ``easy_download`` folder is also accepted.
    background_label : the non-target annotation label (default ``"other"``); every
        other label is treated as a T-cell (sub)type.
    sample_id : optional identifier stamped into both tables as a ``sample_id`` column,
        for later multi-sample concatenation.
    """
    root = Path(easy_download_path)
    if not (root / NETWORK_JSON).is_file() and (root / "harreman_outputs" / NETWORK_JSON).is_file():
        root = root / "harreman_outputs"
    if not (root / NETWORK_JSON).is_file():
        raise FileNotFoundError(f"{NETWORK_JSON} not found under {root}")

    network = json.loads((root / NETWORK_JSON).read_text())
    pair2metab = _pair_to_metabolites(network)
    tiers = _discover_tiers(root)

    master = _build_master(root, network, pair2metab, tiers, background_label)
    genepairs = _build_genepairs(root, pair2metab, tiers, background_label)

    if sample_id is not None:
        master.insert(0, "sample_id", sample_id)
        genepairs.insert(0, "sample_id", sample_id)
    return master, genepairs


def _build_master(root, network, pair2metab, tiers, background_label) -> pd.DataFrame:
    # ---- network baseline: one row per metabolite, gene-pair counts ------------------
    mpc = network.get("metabolite_pair_counts", {})
    genes_per_metab = {
        m: sorted({g for pair in info.get("gene_pair", []) for g in pair})
        for m, info in network.get("gp_per_metabolite", {}).items()
    }
    rows = []
    for metab in sorted(set(mpc) | set(genes_per_metab)):
        rows.append(
            {
                "metabolite": metab,
                "n_gene_pairs": mpc.get(metab, 0),
                "transporter_genes": ", ".join(genes_per_metab.get(metab, [])),
            }
        )
    master = pd.DataFrame(rows).set_index("metabolite")

    # ---- cell-type-independent metabolite significance -------------------------------
    m = _read_cell_indep_m(root)
    master["global_significant"] = master.index.map(m["selected"]).fillna(False).astype(bool)
    master["global_FDR_np"] = master.index.map(m["FDR_np"])

    # how many of a metabolite's gene pairs are significant globally
    sig_pairs_global = _read_sig_gene_pairs(root / CELL_INDEP_GP)
    sig_metab_counts = _count_sig_pairs_per_metabolite(sig_pairs_global, pair2metab)
    master["n_sig_gene_pairs_global"] = (
        master.index.map(sig_metab_counts).fillna(0).astype(int)
    )

    # ---- per tier --------------------------------------------------------------------
    for tier in tiers:
        tm = _read_ct_m(root / tier / CT_M)
        cols = _tier_metabolite_columns(tm, tier, background_label)
        master = master.join(cols)
        # fill sensible defaults for metabolites absent from this tier's table
        master[f"{tier}_n_sig_pairs"] = master[f"{tier}_n_sig_pairs"].fillna(0).astype(int)
        for suffix in ("interactions", "within_Tcell", "Tcell_interfaces"):
            master[f"{tier}_{suffix}"] = master[f"{tier}_{suffix}"].fillna("")
        master[f"{tier}_tcell_involved"] = master[f"{tier}_tcell_involved"] == True  # NaN -> False

    return master.reset_index()


def _tier_metabolite_columns(tm: pd.DataFrame, tier: str, background_label: str) -> pd.DataFrame:
    """For one tier, produce fixed per-metabolite columns summarizing significant
    (undirected) cell-type interactions.

    The CT1/CT2 ordering is not a direction (see module docstring), so an off-diagonal
    pair is rendered as an undirected interface ``A--B`` and a diagonal as ``A (self)``.
    """
    sig = tm[_as_bool(tm["selected"])].copy()
    out = defaultdict(dict)  # metabolite -> {col: val}
    for metab, grp in sig.groupby("metabolite"):
        within, interfaces, allpairs = [], [], []
        tcell_involved = False
        for _, r in grp.iterrows():
            a, b = r["Cell Type 1"], r["Cell Type 2"]
            a_t, b_t = _is_tcell(a, background_label), _is_tcell(b, background_label)
            involves_t = a_t or b_t
            tcell_involved = tcell_involved or involves_t
            if a == b:                                        # diagonal: within one type
                label = f"{a} (self) ({_fmt_fdr(r['FDR_np'])})"
                if a_t:
                    within.append((r["FDR_np"], label))
            else:                                             # off-diagonal: undirected interface
                pair = DASH.join(sorted([a, b]))
                label = f"{pair} ({_fmt_fdr(r['FDR_np'])})"
                if involves_t:
                    interfaces.append((r["FDR_np"], label))
            allpairs.append((involves_t, r["FDR_np"], label))
        # T-cell-involving pairs first, then by FDR
        allpairs.sort(key=lambda t: (not t[0], t[1]))
        within.sort(); interfaces.sort()
        out[metab] = {
            f"{tier}_n_sig_pairs": len(grp),
            f"{tier}_interactions": "; ".join(lbl for _, _, lbl in allpairs),
            f"{tier}_within_Tcell": "; ".join(lbl for _, lbl in within),
            f"{tier}_Tcell_interfaces": "; ".join(lbl for _, lbl in interfaces),
            f"{tier}_tcell_involved": bool(tcell_involved),
        }
    return pd.DataFrame.from_dict(out, orient="index")


def _build_genepairs(root, pair2metab, tiers, background_label) -> pd.DataFrame:
    """Long table: one row per significant gene-pair interaction, across tiers (+ global)."""
    frames = []

    # cell-type-independent gene pairs (tier = 'global', no cell types)
    g = _read_sig_gene_pairs(root / CELL_INDEP_GP)
    if not g.empty:
        g = g.assign(tier="global", cell_type_1=pd.NA, cell_type_2=pd.NA)
        frames.append(g)

    # per-tier gene pairs (CT1/CT2 are sorted labels, NOT a direction — see module docstring)
    for tier in tiers:
        gt = _read_ct_gene_pairs(root / tier / CT_GP)
        if gt.empty:
            continue
        gt = gt.rename(columns={"Cell Type 1": "cell_type_1", "Cell Type 2": "cell_type_2"})
        gt["tier"] = tier
        frames.append(gt)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # attach metabolite(s) and (undirected) T-cell involvement
    out["metabolites"] = out.apply(
        lambda r: " | ".join(sorted(set(pair2metab.get(frozenset((r["Gene 1"], r["Gene 2"])), [])))),
        axis=1,
    )
    out["tcell_involvement"] = out.apply(
        lambda r: _tcell_involvement(r["cell_type_1"], r["cell_type_2"], background_label), axis=1
    )
    keep = [
        "tier", "cell_type_1", "cell_type_2", "Gene 1", "Gene 2",
        "tcell_involvement", "metabolites", "FDR_np", "pval_np", "C_np",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].rename(columns={"Gene 1": "gene1", "Gene 2": "gene2"})
    return out.sort_values(["tier", "tcell_involvement", "FDR_np"]).reset_index(drop=True)


def _tcell_involvement(ct1, ct2, background_label) -> str:
    """Undirected T-cell involvement label for a (CT1, CT2) pair. The ordering is not a
    direction (see module docstring), so we only report set-membership relationships."""
    if not isinstance(ct1, str) or not isinstance(ct2, str):
        return "cell_type_independent"
    t1, t2 = _is_tcell(ct1, background_label), _is_tcell(ct2, background_label)
    if ct1 == ct2:
        return "within_Tcell" if t1 else "within_other"
    if t1 or t2:
        return "Tcell_interface"
    return "non_Tcell_interface"


# ======================================================================================
# CSV readers (normalize column names / metabolite index)
# ======================================================================================
def _read_cell_indep_m(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / CELL_INDEP_M, index_col=0)
    df = df.rename(columns={"Metabolite": "metabolite"})
    df["selected"] = _as_bool(df["selected"])
    return df.set_index("metabolite")


def _read_ct_m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={"Metabolite": "metabolite"})
    df["selected"] = _as_bool(df["selected"])
    return df


def _read_sig_gene_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df["selected"] = _as_bool(df["selected"])
    return df[df["selected"]].copy()


def _read_ct_gene_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df["selected"] = _as_bool(df["selected"])
    return df[df["selected"]].copy()


def _count_sig_pairs_per_metabolite(sig_pairs: pd.DataFrame, pair2metab) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if sig_pairs.empty:
        return counts
    for _, r in sig_pairs.iterrows():
        for metab in pair2metab.get(frozenset((r["Gene 1"], r["Gene 2"])), []):
            counts[metab] += 1
    return counts


# ======================================================================================
# CLI
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="Path to easy_download or harreman_outputs folder")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="Directory to write CSVs (default: <path>/summary)")
    ap.add_argument("--background-label", default="other")
    ap.add_argument("--sample-id", default=None)
    args = ap.parse_args()

    master, genepairs = summarize_harreman_folder(
        args.path, background_label=args.background_label, sample_id=args.sample_id
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.path) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_dir / "metabolite_summary.csv", index=False)
    genepairs.to_csv(out_dir / "gene_pair_summary.csv", index=False)
    print(f"Wrote {len(master)} metabolites -> {out_dir / 'metabolite_summary.csv'}")
    print(f"Wrote {len(genepairs)} gene-pair rows -> {out_dir / 'gene_pair_summary.csv'}")


if __name__ == "__main__":
    main()
