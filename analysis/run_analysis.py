#!/usr/bin/env python
"""Phase 2 offline deterministic analysis (RQ1-RQ4). Reads data/ and results/, writes
results/, paper/figures/, paper/generated/. See analysis/PLAN.md (pre-registered).

Objective subset rules used for figures/tables (documented per PLAN deviation note):
- capacity table/heatmap models: dedup-representative rows, params known, released >= 2024-01-01,
  sorted by kv_bytes_per_token_bf16 descending, first N; plus the 2023 dedup representative with
  the LOWEST kv-per-param (the trend-story anchor).
- validation: all rows of results/kv_validation.csv.
"""
import os
os.environ["SOURCE_DATE_EPOCH"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, math, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import kv as kvm

GIB = 2 ** 30
YEARS = [2023, 2024, 2025, 2026]
YLET = {2023: "TwentyThree", 2024: "TwentyFour", 2025: "TwentyFive", 2026: "TwentySix"}

def load():
    d = pd.read_csv(ROOT / "data/models.csv")
    d["year"] = d.release_date.str[:4].astype(int)
    d["kv_tok"] = d.kv_bytes_per_token_bf16.astype(float)
    d["params"] = d.total_params_b.astype(float)  # NaN allowed
    d["kv_per_param"] = d.kv_tok / (d.params * 1e9)
    # dedup: earliest release_date per group, tie lexicographic repo_id
    dd = d.sort_values(["release_date", "repo_id"], kind="mergesort")
    dedup = dd.groupby("dedup_group", sort=True).head(1).copy()
    return d, dedup

def coarse_class(label: str) -> str:
    s = str(label).lower()
    if "mla" in s: return "MLA latent"
    if any(t in s for t in ("swa", "window", "local")): return "windowed"
    if any(t in s for t in ("hybrid", "mamba", "rwkv", "gdn", "lightning", "kda", "recurrent")): return "recurrent/hybrid"
    return "full attention"

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "--"
    return f"{x:.{nd}f}"

def fmt_gib(x, nd=1):
    if x is None or not np.isfinite(x): return "--"
    return f"{x/GIB:.{nd}f}"

macros, tables = {}, []

def M(name, value):
    macros[name] = str(value)

# ================================================================ RQ1
def rq1(dedup):
    out = {}
    for y in YEARS:
        s = dedup[dedup.year == y]
        out[y] = dict(
            n=len(s),
            med_kv_tok=float(np.median(s.kv_tok)),
            iqr_kv_tok=float(np.subtract(*np.percentile(s.kv_tok, [75, 25]))),
            med_pp=float(np.nanmedian(s.kv_per_param)),
            iqr_pp=float(np.subtract(*np.nanpercentile(s.kv_per_param, [75, 25]))),
        )
    import datetime as _dt
    date_ord = pd.to_datetime(dedup.release_date).map(_dt.datetime.toordinal).astype(float).values
    rho, p = stats.spearmanr(date_ord, dedup.kv_tok.values)
    ok = np.isfinite(dedup.kv_per_param.values)
    rho_pp, p_pp = stats.spearmanr(date_ord[ok], dedup.kv_per_param.values[ok])
    m23 = out[2023]["med_pp"]
    m_new = float(np.nanmedian(pd.concat([dedup[dedup.year == 2025], dedup[dedup.year == 2026]]).kv_per_param))
    ratio = m_new / m23
    verdict = "supported" if ratio <= 0.75 else ("partial" if ratio <= 0.90 else "refuted")
    return out, dict(rho=rho, p=p, rho_pp=rho_pp, p_pp=p_pp, ratio=ratio, m23=m23, m_new=m_new, verdict=verdict)

# ================================================================ RQ2
def ols(X, y):
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
    yhat = beta[0] + X @ beta[1:]
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return beta, 1 - ss_res / ss_tot

def rq2(dedup):
    s = dedup[dedup.params.notna() & (dedup.kv_tok > 0)].copy()
    n_zero = int((dedup.kv_tok == 0).sum())
    y = np.log10(s.kv_tok.values)
    xp = np.log10(s.params.values)
    beta, r2 = ols(xp.reshape(-1, 1), y)
    rng = np.random.default_rng(0)
    boots = []
    idx = np.arange(len(s))
    for _ in range(1000):
        b = rng.choice(idx, size=len(idx), replace=True)
        try: _, r2b = ols(xp[b].reshape(-1, 1), y[b]); boots.append(r2b)
        except Exception: pass
    ci = np.percentile(boots, [2.5, 97.5])
    h2 = "weak" if r2 < 0.5 else "not weak"

    s["gqa"] = s.num_attention_heads / s.num_kv_heads
    feats = dict(
        params=np.log10(s.params.values),
        gqa=np.log1p(s.gqa.values),
        head_dim=s.head_dim.values.astype(float),
        depth=np.log1p(s.num_layers.values.astype(float)),
        window_share=(s.n_window_layers / s.num_layers).values.astype(float),
        recurrent_share=(s.n_recurrent_layers / s.num_layers).values.astype(float),
        latent=(s.n_latent_layers > 0).astype(float).values,
        shared=s.kv_shared_cross_layer.astype(str).str.lower().eq("true").astype(float).values,
    )
    F = pd.DataFrame(feats).fillna(0.0)
    keep = F.columns[F.std() > 0]
    F = F[keep]
    z = lambda a: (a - a.mean()) / (a.std() or 1)
    Z = np.column_stack([z(F[c].values) for c in F.columns])
    bz, r2_full = ols(Z, y)
    cols_geom = [c for c in F.columns if c != "params"]
    Zg = np.column_stack([z(F[c].values) for c in cols_geom])
    _, r2_geom = ols(Zg, y)
    _, r2_par = ols(z(F.params.values).reshape(-1, 1), y)
    vif = {}
    for c in F.columns:
        others = [o for o in F.columns if o != c]
        if not others: vif[c] = 1.0; continue
        _, r2j = ols(F[others].values, F[c].values)
        vif[c] = 1 / max(1e-12, 1 - r2j)
    moe = s.moe.astype(str).str.lower().eq("true")
    a = s[~moe].kv_per_param.dropna(); b_ = s[moe].kv_per_param.dropna()
    u, pmw = stats.mannwhitneyu(a, b_, alternative="two-sided")
    rb = 1 - 2 * u / (len(a) * len(b_))
    med_dense, med_moe = float(np.median(a)), float(np.median(b_))
    h3_ratio = med_moe / med_dense
    h3 = "supported" if h3_ratio <= 0.5 else "refuted"
    return dict(n=len(s), n_zero=n_zero, beta=beta, r2=r2, ci=ci, h2=h2, betas=dict(zip(F.columns, bz)),
                r2_full=r2_full, r2_geom=r2_geom, r2_par=r2_par, vif=vif,
                moe=dict(n_dense=len(a), n_moe=len(b_), med_dense=med_dense, med_moe=med_moe,
                         p=float(pmw), rb=float(rb), ratio=h3_ratio, verdict=h3))

# ================================================================ RQ3
WPREC = {"bf16": 2.0, "fp8": 1.0, "4bit": 4.5 / 8}
KPREC = {"bf16": 2, "fp8": 1}
BUDGETS = {"24": 24 * GIB, "80": 80 * GIB, "128": 128 * GIB}
OVERHEADS = {"0": 0, "2": 2 * GIB, "6": 6 * GIB, "12": 12 * GIB}

def model_curves(repo_id, bytes_elem):
    cfg = json.loads((ROOT / "data/raw/configs" / (repo_id.replace("/", "__") + ".json")).read_text())
    layers = kvm.parse_layers(cfg, bytes_per_element=bytes_elem)
    S, full_rate, latent_rate, wins = 0.0, 0.0, 0.0, []
    for L in layers:
        S += L.fixed_state_bytes
        if L.kind == "full": full_rate += 2 * L.n_kv_heads * L.head_dim * L.bytes_per_element
        elif L.kind == "latent": latent_rate += (L.latent_elems or 0) * L.bytes_per_element
        elif L.kind == "window": wins.append((2 * L.n_kv_heads * L.head_dim * L.bytes_per_element, L.window))
    return S, full_rate, latent_rate, sorted(wins, key=lambda t: t[1])

def total_bytes(curve, T):
    S, fr, lr, wins = curve
    tot = S + (fr + lr) * T
    for rate, W in wins: tot += rate * min(T, W)
    return tot

def solve_tokens(curve, pool):
    """Exact piecewise-linear solve: smallest T with total(T) >= pool; inf if asymptote < pool."""
    S, fr, lr, wins = curve
    if fr + lr <= 0:
        cap = S + sum(r * W for r, W in wins)
        return math.inf if cap <= pool else _solve_bp(curve, pool)
    return _solve_bp(curve, pool)

def _solve_bp(curve, pool):
    S, fr, lr, wins = curve
    bps = sorted({W for _, W in wins})
    prev = 0.0
    for W in bps:
        # segment (prev, W]: rate = fr+lr+sum(rate_w for W_w > prev ... still growing at T in (prev,W])
        rate = fr + lr + sum(r for r, Ww in wins if Ww > prev)
        t0 = total_bytes(curve, prev)
        if rate > 0 and t0 + rate * (W - prev) >= pool and t0 < pool:
            return prev + (pool - t0) / rate
        prev = W
    rate = fr + lr
    t0 = total_bytes(curve, prev)
    if rate <= 0: return math.inf if t0 < pool else prev
    return prev + max(0.0, (pool - t0)) / rate

def rq3(dedup):
    s = dedup[dedup.params.notna()].copy()
    s["Tstar"] = np.nan; s["Tstar_inf"] = False
    curves = {}
    for repo in s.repo_id:
        cfg = json.loads((ROOT / "data/raw/configs" / (repo.replace("/", "__") + ".json")).read_text())
        T = kvm.kv_par_context(cfg, total_params=float(s.loc[s.repo_id == repo, "params"].iloc[0]) * 1e9,
                               weight_bytes_per_param=2.0, bytes_per_element=2)
        s.loc[s.repo_id == repo, "Tstar"] = T
        if not np.isfinite(T): s.loc[s.repo_id == repo, "Tstar_inf"] = True
    for kp, belem in KPREC.items():
        curves[kp] = {repo: model_curves(repo, belem) for repo in s.repo_id}
    rows = []
    for r in s.itertuples():
        params_b = float(r.params)
        for wp, wbpp in WPREC.items():
            wbytes = params_b * 1e9 * wbpp
            for bp, budget in BUDGETS.items():
                for ov in OVERHEADS:
                    pool = budget - wbytes - OVERHEADS[ov]
                    for kp in KPREC:
                        cur = curves[kp][r.repo_id]
                        if pool <= 0:
                            rows.append(dict(repo_id=r.repo_id, year=r.year, params_b=params_b,
                                             wprec=wp, kvprec=kp, budget_gib=bp, overhead_gib=ov,
                                             max_tokens=None, conc_32k=0, conc_128k=0))
                            continue
                        mt = solve_tokens(cur, pool)
                        c32 = int(pool // total_bytes(cur, 32768)) if total_bytes(cur, 32768) <= pool else 0
                        c128 = int(pool // total_bytes(cur, 131072)) if total_bytes(cur, 131072) <= pool else 0
                        rows.append(dict(repo_id=r.repo_id, year=r.year, params_b=params_b,
                                         wprec=wp, kvprec=kp, budget_gib=bp, overhead_gib=ov,
                                         max_tokens=mt,
                                         conc_32k=c32, conc_128k=c128))
    cap = pd.DataFrame(rows)
    per_year = {}
    for y in YEARS:
        sy = s[s.year == y]
        fin = sy[~sy.Tstar_inf].Tstar
        per_year[y] = dict(n=len(sy), med_Tstar=float(np.median(fin)) if len(fin) else float("nan"),
                           iqr_Tstar=float(np.subtract(*np.percentile(fin, [75, 25]))) if len(fin) else float("nan"),
                           inf_share=float(sy.Tstar_inf.mean()))
    r_23 = per_year[2023]["med_Tstar"]
    r_new = float(np.nanmedian(s[(s.year >= 2025)].Tstar[~s[s.year >= 2025].Tstar_inf]))
    h4 = "supported" if (np.isfinite(r_23) and r_new >= 10 * r_23) else "refuted"
    dom = {y: float((s[s.year == y].Tstar < 32768).mean()) for y in YEARS}
    return s, cap, per_year, dict(ratio=(r_new / r_23 if r_23 else float("nan")), verdict=h4, dom=dom)

# ================================================================ main
def main():
    d, dedup = load()
    M("CensusRows", len(d)); M("DedupRows", len(dedup)); M("OrgCount", d.org.nunique())
    M("Groups", d.dedup_group.nunique())
    gated = len((ROOT / "data/gated_skipped.txt").read_text().split())
    M("GatedSkipped", gated)
    M("KvErrRows", len((ROOT / "data/kv_config_errors.txt").read_text().splitlines()))
    M("NoParams", int(d.params.isna().sum()))
    for y in YEARS: M(f"YearRows{YLET[y]}", int((d.year == y).sum()))

    r1, r1s = rq1(dedup)
    for y in YEARS:
        M(f"MedianKvTok{YLET[y]}", fmt(r1[y]["med_kv_tok"], 0))
        M(f"MedianKvpp{YLET[y]}", fmt(r1[y]["med_pp"] * 1e9, 1))  # bytes/token per PARAM, scaled 1e9 for readability
        M(f"RqOneN{YLET[y]}", r1[y]["n"])
    M("SpearmanRho", fmt(r1s["rho"], 3)); M("SpearmanP", f"{r1s['p']:.1e}")
    M("SpearmanRhoPP", fmt(r1s["rho_pp"], 3)); M("SpearmanPPp", f"{r1s['p_pp']:.1e}")
    M("HOneRatio", fmt(r1s["ratio"], 3)); M("HOneVerdict", r1s["verdict"])

    r2 = rq2(dedup)
    M("RegN", r2["n"]); M("RegZeroKV", r2["n_zero"])
    M("ParamsSlope", fmt(r2["beta"][1], 3)); M("ParamsRsq", fmt(r2["r2"], 3))
    M("ParamsRsqLo", fmt(r2["ci"][0], 3)); M("ParamsRsqHi", fmt(r2["ci"][1], 3))
    M("HTwoVerdict", r2["h2"])
    M("GeomRsq", fmt(r2["r2_geom"], 3)); M("ParRsq", fmt(r2["r2_par"], 3)); M("FullRsq", fmt(r2["r2_full"], 3))
    for k, v in r2["betas"].items(): M("Beta" + "".join(w.capitalize() for w in k.split("_")), fmt(v, 3))
    for k, v in r2["vif"].items(): M("Vif" + "".join(w.capitalize() for w in k.split("_")), fmt(v, 1))
    M("MoeN", r2["moe"]["n_moe"]); M("DenseN", r2["moe"]["n_dense"])
    M("MedKvppDense", fmt(r2["moe"]["med_dense"] * 1e9, 1)); M("MedKvppMoe", fmt(r2["moe"]["med_moe"] * 1e9, 1))
    M("MoeP", f"{r2['moe']['p']:.1e}"); M("MoeRB", fmt(r2["moe"]["rb"], 3))
    M("HThreeRatio", fmt(r2["moe"]["ratio"], 3)); M("HThreeVerdict", r2["moe"]["verdict"])

    s3, cap, per_year, r3s = rq3(dedup)
    for y in YEARS:
        M(f"ParityMed{YLET[y]}", fmt(per_year[y]["med_Tstar"], 0))
        M(f"ParityInfShare{YLET[y]}", fmt(100 * per_year[y]["inf_share"], 1))
        M(f"DominantShare{YLET[y]}", fmt(100 * r3s["dom"][y], 1))
    M("HFourRatio", fmt(r3s["ratio"], 1)); M("HFourVerdict", r3s["verdict"])

    # headline capacity: representative subset (objective rule, PLAN deviation note):
    # collapse dedup rows by (org, num_layers, num_kv_heads, head_dim, attention_type, window_size)
    # keeping the earliest release; subset = 2023 min-kv/param anchor + top-12 by params +
    # top-12 by kv/token among collapsed rows with year >= 2024 (params known), then order by
    # kv_tok descending for the figure.
    dd = dedup[dedup.params.notna()].copy()
    sig = dd.groupby(["org", "num_layers", "num_kv_heads", "head_dim", "attention_type", "window_size"],
                     dropna=False, sort=True).head(1)
    anchor = dedup[(dedup.params.notna()) & (dedup.year == 2023)].sort_values("kv_per_param").head(1)
    feas = cap[(cap.budget_gib == "128") & (cap.wprec == "4bit") & (cap.kvprec == "bf16") & (cap.overhead_gib == "6")]
    n_infeas = int(feas.max_tokens.isna().sum())
    M("InfeasibleBigModels", n_infeas)
    feasible_ids = set(feas[feas.max_tokens.notna()].repo_id)
    recent = sig[(sig.year >= 2024) & (sig.repo_id.isin(feasible_ids))]
    sub = list(anchor.repo_id) + list(recent.sort_values("params", ascending=False).head(12).repo_id) \
        + list(recent.sort_values("kv_tok", ascending=False).head(12).repo_id)
    sub = list(dict.fromkeys(sub))
    heat = cap[(cap.repo_id.isin(sub)) & (cap.overhead_gib == "6") & (cap.kvprec == "bf16")]

    # RQ4 validation
    vrows = list(csv.DictReader((ROOT / "results/kv_validation.csv").open()))
    within = sum(1 for r in vrows if abs(float(r["rel_error"])) <= 0.01)
    M("ValRows", len(vrows)); M("ValClasses", len({r["attention_class"] for r in vrows})); M("ValWithin", within)

    # ---------------- outputs: results/
    pd.DataFrame([dict(year=y, **{k: v for k, v in r1[y].items()}) for y in YEARS]).to_csv(ROOT / "results/rq1_stats.csv", index=False)
    json.dump(dict(rq1_spearman={k: (v if isinstance(v, str) else float(v)) for k, v in r1s.items()},
                   rq2={k: v for k, v in r2.items() if k not in ("beta", "ci")},
                   rq2_ci=list(map(float, r2["ci"])), rq2_slope=float(r2["beta"][1]),
                   rq3_parity=per_year, rq3_h4=r3s),
              (ROOT / "results/analysis_stats.json").open("w"), indent=1, default=float)
    cap[cap.overhead_gib == "6"].to_csv(ROOT / "results/rq3_capacity_overhead6.csv", index=False)
    cap[(cap.budget_gib == "128") & (cap.wprec == "bf16") & (cap.kvprec == "bf16")].to_csv(ROOT / "results/rq3_sensitivity_overhead.csv", index=False)
    dedup.to_csv(ROOT / "results/dedup_rows.csv", index=False)

    make_figures(d, dedup, s3, heat, sub, vrows, r2)
    make_tex(d, dedup, r1, r1s, r2, s3, cap, per_year, r3s, heat, sub, vrows)
    print("analysis complete")

# ================================================================ figures
def make_figures(d, dedup, s3, heat, sub, vrows, r2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MD = {"CreationDate": None, "ModDate": None, "Creator": "kv-census", "Producer": "kv-census"}
    FIG = ROOT / "paper/figures"
    dedup = dedup.copy(); dedup["cls"] = dedup.attention_type.map(coarse_class)
    colors = {"full attention": "#1f77b4", "windowed": "#2ca02c", "recurrent/hybrid": "#9467bd", "MLA latent": "#d62728"}

    def save(fig, name):
        fig.savefig(FIG / name, format="pdf", metadata=MD, bbox_inches="tight")
        plt.close(fig)

    # fig1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, g in dedup.groupby("cls"):
        ax.scatter(pd.to_datetime(g.release_date), g.kv_tok.clip(lower=1), s=9, alpha=.6, label=f"{c} (n={len(g)})", c=colors.get(c, "k"))
    ax.set_yscale("log"); ax.set_ylabel("KV bytes/token (BF16, marginal)")
    ax.set_xlabel("release date (Hub createdAt)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig1_kv_trend.pdf")

    # fig2
    s = dedup[dedup.params.notna() & (dedup.kv_tok > 0)]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for c, g in s.groupby("cls"):
        ax.scatter(g.params, g.kv_tok, s=9, alpha=.6, label=c, c=colors.get(c, "k"))
    xs = np.logspace(np.log10(s.params.min()), np.log10(s.params.max()), 50)
    b = r2["beta"]
    ax.plot(xs, 10 ** (b[0] + b[1] * np.log10(xs)), "k--", lw=1, label=f"OLS fit, $R^2$={r2['r2']:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total parameters"); ax.set_ylabel("KV bytes/token (BF16)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig2_kv_vs_params.pdf")

    # fig3
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    vals = [r2["r2_par"], r2["r2_geom"], r2["r2_full"]]
    ax.bar(["params\nonly", "geometry &\nmechanisms only", "full model"], vals, color=["#1f77b4", "#2ca02c", "#444444"])
    for i, v in enumerate(vals): ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("$R^2$ (log10 KV bytes/token)"); ax.set_ylim(0, 1)
    save(fig, "fig3_mechanism.pdf")

    # fig4 heatmap
    piv = heat.pivot_table(index="repo_id", columns=["budget_gib", "wprec"], values="max_tokens")
    piv = piv.reindex([r for r in sub if r in piv.index])
    cols = [("24", "bf16"), ("24", "fp8"), ("24", "4bit"), ("128", "bf16"), ("128", "fp8"), ("128", "4bit")]
    piv = piv[[c for c in cols if c in piv.columns]]
    data = np.log10(piv.astype(float).fillna(0).clip(lower=1))
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(piv) + 1.6))
    im = ax.imshow(data.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{b} GiB / {w}" for b, w in piv.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([r.split("/")[-1].replace("_", r"\_") for r in piv.index], fontsize=7)
    cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label(r"$\log_{10}$ max KV tokens", fontsize=8)
    save(fig, "fig4_capacity_heatmap.pdf")

    # fig5 parity trend
    ys = [2023, 2024, 2025, 2026]
    med = [np.nanmedian(s3[s3.year == y].Tstar[~s3[s3.year == y].Tstar_inf]) for y in ys]
    lo = [np.percentile(s3[s3.year == y].Tstar[~s3[s3.year == y].Tstar_inf], 25) for y in ys]
    hi = [np.percentile(s3[s3.year == y].Tstar[~s3[s3.year == y].Tstar_inf], 75) for y in ys]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.fill_between(ys, lo, hi, alpha=.25, label="IQR (finite values)")
    ax.plot(ys, med, "o-", label="median")
    ax.set_yscale("log"); ax.set_xticks(ys)
    ax.set_xlabel("release year"); ax.set_ylabel("KV-parity context $T^*$ (tokens)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig5_parity_trend.pdf")

    # fig6 validation
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    a = [float(r["analytic_bytes"]) for r in vrows]; m = [float(r["measured_bytes"]) for r in vrows]
    ax.scatter(a, m, s=28, c="#d62728")
    lim = [min(a + m) * .8, max(a + m) * 1.25]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("analytic bytes (kv.py)"); ax.set_ylabel("measured bytes (HF cache)")
    save(fig, "fig6_validation.pdf")

# ================================================================ tex
def esc(s): return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

def make_tex(d, dedup, r1, r1s, r2, s3, cap, per_year, r3s, heat, sub, vrows):
    GEN = ROOT / "paper/generated"
    lines = ["% AUTO-GENERATED by make analysis - do not edit by hand"]
    for k in sorted(macros): lines.append(rf"\newcommand{{\kvc{k}}}{{{macros[k]}}}")
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n")

    T = []
    T.append(r"% AUTO-GENERATED by make analysis - do not edit by hand")
    # dataset overview
    T.append(r"\begin{table}[t]\centering\small" "\n" r"\caption{Census overview. Rows are checkpoints; the deduplicated set keeps one row per geometry group.}\label{tab:overview}")
    T.append(r"\begin{tabular}{lrrrrr}\toprule")
    T.append(r"year & rows & dedup rows & median KV B/tok & median KV B/tok per $10^9$ params\\\midrule")
    for y in YEARS:
        T.append(f"{y} & {r1[y]['n']} & {r1[y]['n']} & {fmt(r1[y]['med_kv_tok'],0)} & {fmt(r1[y]['med_pp']*1e9,1)}\\\\")
    T.append(r"\bottomrule\end{tabular}\end{table}")
    (GEN / "table_overview.tex").write_text("\n".join(T) + "\n")

    T2 = [r"% AUTO-GENERATED"]
    T2.append(r"\begin{table}[t]\centering\small" "\n" r"\caption{Standardized coefficients of the mechanism regression on $\log_{10}$ KV bytes/token.}\label{tab:mechanism}")
    T2.append(r"\begin{tabular}{lrr}\toprule predictor & $\beta$ (std.) & VIF\\\midrule")
    for k in r2["betas"]:
        nm = {"params": "total params ($\\log_{10}$)", "gqa": "GQA ratio ($\\log(1+r)$)", "head_dim": "head dim",
              "depth": "depth ($\\log(1+L)$)", "window_share": "window layer share", "recurrent_share": "recurrent layer share",
              "latent": "MLA latent (0/1)", "shared": "cross-layer shared KV (0/1)"}.get(k, k)
        T2.append(f"{nm} & {fmt(r2['betas'][k],3)} & {fmt(r2['vif'][k],1)}\\\\")
    T2.append(r"\midrule")
    T2.append(f"$N$ & \\multicolumn{{2}}{{r}}{{{r2['n']}}}\\\\")
    T2.append(f"$R^2$ (full) & \\multicolumn{{2}}{{r}}{{{fmt(r2['r2_full'],3)}}}\\\\")
    T2.append(r"\bottomrule\end{tabular}\end{table}")
    (GEN / "table_mechanism.tex").write_text("\n".join(T2) + "\n")

    T3 = [r"% AUTO-GENERATED"]
    T3.append(r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}" "\n" r"\caption{Maximum KV tokens at 6 GiB runtime overhead, KV in BF16; \lq--\rq{} means the weights alone exceed the budget, $\infty$ means the cache saturates (bounded window/state).}\label{tab:capacity}")
    T3.append(r"\begin{tabular}{lrrrrrr}\toprule")
    T3.append(r"model & 24/BF16 & 24/FP8 & 24/4b & 128/BF16 & 128/FP8 & 128/4b\\\midrule")
    h = heat.copy()
    for repo in [r for r in sub if r in set(h.repo_id)]:
        cells = []
        for c in [("24","bf16"),("24","fp8"),("24","4bit"),("128","bf16"),("128","fp8"),("128","4bit")]:
            row = h[(h.repo_id == repo) & (h.budget_gib == c[0]) & (h.wprec == c[1])]
            v = row.max_tokens.iloc[0] if len(row) else None
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("--")
            elif math.isinf(float(v)):
                cells.append(r"$\infty$")
            elif v >= 1e6:
                cells.append(f"{v/1e6:.1f}M")
            else:
                cells.append(f"{v/1e3:.0f}k")
        T3.append(f"{esc(repo.split('/')[-1])} & " + " & ".join(cells) + r"\\")
    T3.append(r"\bottomrule\end{tabular}\end{table}")
    (GEN / "table_capacity.tex").write_text("\n".join(T3) + "\n")

    T4 = [r"% AUTO-GENERATED"]
    T4.append(r"\begin{table}[t]\centering\small" "\n" r"\caption{RQ4 validation: analytic (kv.py) vs measured HF cache bytes on reduced-depth random-init instances.}\label{tab:validation}")
    T4.append(r"\begin{tabular}{llrrrl}\toprule model & class & analytic & measured & rel.err & note\\\midrule")
    for r in vrows:
        note = esc(r["explanation"][:60]) + ("" if len(r["explanation"]) <= 60 else r"\ldots")
        T4.append(f"{esc(r['model_or_arch'].split('/')[-1])} & {esc(r['attention_class'])} & {float(r['analytic_bytes'])/1e6:.2f}M & {float(r['measured_bytes'])/1e6:.2f}M & {100*float(r['rel_error']):.3f}\\% & {note}\\\\")
    T4.append(r"\bottomrule\end{tabular}\end{table}")
    (GEN / "table_validation.tex").write_text("\n".join(T4) + "\n")
    print("tex written")

if __name__ == "__main__":
    main()
