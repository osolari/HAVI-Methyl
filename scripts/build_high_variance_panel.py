"""Phase 5 helper: build a per-CpG panel where the prediction problem is well-posed.

The Loyfer U25 tissue-marker panel is wrong for the FinaleMe paired-data
bench (tissue-discriminating, low cross-patient variance, see Phase 5
notes). This script scans the Liu 2024 ``meth_wgbs/*.bed.gz`` files once,
keeps CpGs that pass coverage and presence filters, and writes the top-N
by cross-patient methylation variance to a BED panel.

Resulting panel is suitable input to ``bench_finaleme_realdata.py
--locus-panel`` because:
  - every CpG actually varies across patients (Pearson r is identifiable),
  - every CpG has coverage in most samples (no NaN sinks).
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np


def _norm_chrom(c: str) -> str:
    return c if c.startswith("chr") else f"chr{c}"


def _scan_one(path: Path, chroms_allow: set[str]) -> dict[tuple[str, int], tuple[float, int]]:
    """Yield {(chrom, start): (meth_pct, coverage)} from a Liu 2024 WGBS BED.gz."""
    out: dict[tuple[str, int], tuple[float, int]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("track") or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 8:
                continue
            chrom = _norm_chrom(parts[0])
            if chrom not in chroms_allow:
                continue
            try:
                start = int(parts[1])
                meth_pct = float(parts[6])
                cov = int(parts[7])
            except (ValueError, IndexError):
                continue
            out[(chrom, start)] = (meth_pct, cov)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meth-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--min-cov", type=int, default=5)
    ap.add_argument("--min-presence-frac", type=float, default=0.60)
    ap.add_argument(
        "--chroms",
        default="chr19,chr20,chr21,chr22",
        help="Comma-separated chromosomes to consider (default: chr19..chr22 keeps the scan cheap and the panel well-mixed).",
    )
    args = ap.parse_args()

    meth_dir = Path(args.meth_dir)
    chroms_allow = set(args.chroms.split(","))
    files = sorted(p for p in meth_dir.glob("*.bed.gz") if not p.name.startswith("._"))
    if not files:
        raise SystemExit(f"No WGBS files under {meth_dir}")
    print(f"Scanning {len(files)} WGBS files on {sorted(chroms_allow)} ...")

    meth_per_site: dict[tuple[str, int], list[float]] = defaultdict(list)
    presence: dict[tuple[str, int], int] = defaultdict(int)
    for i, p in enumerate(files):
        per_site = _scan_one(p, chroms_allow)
        kept = 0
        for k, (pct, cov) in per_site.items():
            if cov < args.min_cov:
                continue
            meth_per_site[k].append(pct / 100.0)
            presence[k] += 1
            kept += 1
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"  [{i + 1}/{len(files)}] {p.name}: kept {kept} CpGs (>= {args.min_cov} cov)")

    n_samples = len(files)
    min_presence = int(np.ceil(args.min_presence_frac * n_samples))
    candidates: list[tuple[float, tuple[str, int]]] = []
    for k, vals in meth_per_site.items():
        if presence[k] < min_presence:
            continue
        v = float(np.std(vals))
        candidates.append((v, k))
    print(
        f"{len(candidates)} CpGs pass presence >= {min_presence}/{n_samples} "
        f"and cov >= {args.min_cov}; picking top {args.top_n} by std"
    )
    candidates.sort(reverse=True)
    pick = candidates[: args.top_n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for v, (chrom, start) in sorted(pick, key=lambda x: (x[1][0], x[1][1])):
            f.write(f"{chrom}\t{start}\t{start + 1}\tstd={v:.3f}\n")
    print(f"Wrote {out} with {len(pick)} CpGs")
    if pick:
        stds = [v for v, _ in pick]
        print(
            f"  std (top picks): min={min(stds):.3f} median={np.median(stds):.3f} max={max(stds):.3f}"
        )


if __name__ == "__main__":
    main()
