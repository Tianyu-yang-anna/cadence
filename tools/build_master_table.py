"""Rebuild every comparison table in the reports from the raw metrics JSONs.

The reports quote a lot of numbers across a lot of arms, and hand-copying them
is how transcription errors get in. This reads results/benchgen_*/ directly and
emits the master tables as markdown + a flat CSV, so any number in the write-up
can be traced to a file on disk.

ARM REGISTRY. Every row below names (run dir, TAG, what it is). A row is only
as trustworthy as its provenance note, so the caveats that qualify a number
(non-strict budget, degenerate output, retracted control) travel WITH the row
instead of living in prose the table can drift away from.

Usage:
  python tools/build_master_table.py --out_md docs/reports/MASTER_TABLES.md \
      --out_csv results/master_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHES = ["wikipedia", "wikisource", "tinystories", "lm1b"]
BENCH_LABEL = {"wikipedia": "Wikipedia", "wikisource": "WikiSource",
               "tinystories": "TinyStories", "lm1b": "1BW"}

# (group, label, run_dir, tag, nfe_backbone, nfe_sampler, note)
# nfe = backbone forwards per 1024 generated tokens, both CFG branches counted.
ROWS = [
    # ---- CADENCE, strict 2B, in mechanism order -------------------------
    ("cadence", "基线 sh（无 depth 无 MaskGIT）",
     "benchgen_planner_prefix_owt2_pqsh", "_final", 22, 0, ""),
    ("cadence", "基线 ps（per-scale 码本）",
     "benchgen_planner_prefix_owt2_pqps", "_final", 22, 0, ""),
    ("cadence", "+depth-AR（_finalDA）",
     "benchgen_planner_prefix_owt2_pqsh_b2pl", "_finalDA", 22, 0, ""),
    ("cadence", "+MaskGIT 对齐微调+精化（_finalMG）",
     "benchgen_planner_prefix_owt2_pqsh_b2mg3", "_finalMG", 64, 0, ""),
    ("cadence", "+MaskGIT 合并微调+精化（_finalB2）",
     "benchgen_planner_prefix_owt2_pqsh_b2mgd", "_finalB2", 64, 0, ""),
    ("cadence", "+分阶段课程（_finalSQ）",
     "benchgen_planner_prefix_owt2_pqsh_b2sq2", "_finalSQ", 64, 0, ""),
    ("cadence", "位置轴采样器 MaskGIT（_finalPOS）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalPOS", 22, 48,
     "跑在 b2sp 上，该臂实际训练模式是 sampler_mix 而非 sampler_pos（终章十一 §6.1）"),
    ("cadence", "★段轴采样器 MaskGIT（_finalSEG，主行）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg", "_finalSEG", 22, 88, ""),
    ("cadence", "（参考）final4：4.6B 非严格预算",
     "benchgen_planner_prefix_owt2_pqsh_mgd", "_final4", 64, 0,
     "**不是严格 2B**，多花约 2.6B 微调 token，仅作规模参考"),

    # ---- the three-axis single-variable comparison (2026-09-04) ---------
    ("axis", "段轴 seg:all:4（b2sg）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg", "_finalSEG", 22, 88,
     "depth 冻结"),
    ("axis", "位置轴 pos:8（b2spf，纯 sampler_pos）",
     "benchgen_planner_prefix_owt2_pqsh_b2spf", "_finalPOSF", 22, 48,
     "depth 冻结"),
    ("axis", "chunk 轴 lr:8:1（b2slr3）",
     "benchgen_planner_prefix_owt2_pqsh_b2slr3", "_finalLR3", 22, 16,
     "depth 冻结；C=8 由预注册规则在 sel 上选出，但该选择不可靠（两 sel 集 ρ=−0.90）"),
    ("axis", "（旧）位置轴 pos:8（b2sp，sampler_mix）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalPOS", 22, 48,
     "训练模式污染，见 §6.1；depth 可训。保留以对照"),
    ("axis", "（作废）chunk 轴 lr C16K4（b2sp）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalLR", 22, 128,
     "**作废**：训练/推理不匹配（b2sp 从未训过 lr 揭示模式）"),
    ("axis", "（旧）严格位置 AR ar:8（b2sp）",
     "benchgen_planner_prefix_owt2_pqsh_b2sp", "_finalAR", 22, 0,
     "单尺度 ar:8；sampler_causal 恰是 b2sp 训练的一半，故为匹配较好的一行"),

    # ---- the 2D-maskgit wave (2026-09-04 evening) ------------------------
    ("axis", "2D 赢家 lrseg C=8,K=1（b2s2d，_final2D）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2d", "_final2D", 22, 48,
     "depth 冻结；两 sel 集对 chunk 超参排序反相关，此格是主判据下的名义赢家"),
    ("axis", "纯段轴@细尺度锚 lrseg C=1,K=4（b2s2d，_finalS24）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2d", "_finalS24", 22, 24,
     "= seg:8,9,10:4 的同臂退化（逐比特等价有门禁测试）；24 采样器前向的注册数"),

    # ---- P1 HMAR scale reweighting --------------------------------------
    ("hmar", "★α=0.25 插值全链（sg56a25，_finalSEGA25）",
     "benchgen_planner_prefix_owt2_pqsh_sg56a25", "_finalSEGA25", 22, 88,
     "w∝token^0.25·lognormal^0.75；R1/R2 四集全涨、MAUVE 近平 —— "
     "token 与 lognormal 之间的新帕累托点；门控见 results/hmar_alpha/"),
    ("hmar", "波A token（hw76tk，单跑基座）",
     "benchgen_planner_prefix_owt2_pqsh_hw76tk", "_finalHWTK", 22, 0,
     "非主线：无 MaskGIT，隔离用"),
    ("hmar", "波A equal（hw76eq）",
     "benchgen_planner_prefix_owt2_pqsh_hw76eq", "_finalHWEQ", 22, 0,
     "非主线"),
    ("hmar", "波A lognormal（hw76ln）",
     "benchgen_planner_prefix_owt2_pqsh_hw76ln", "_finalHWLN", 22, 0,
     "非主线"),
    ("hmar", "波B lognormal 全链（sg56ln）",
     "benchgen_planner_prefix_owt2_pqsh_sg56ln", "_finalSEGHWLN", 22, 88,
     "主线对照 = _finalSEG（token 加权）"),

    # ---- fluent baselines, directly comparable --------------------------
    ("baseline", "BD3-LM 满预算", "benchgen_bd3lm_owt2", "_final", 1024, 0, ""),
    ("baseline", "BD3-LM 限步 256（每块 4 步）",
     "benchgen_bd3lm_owt2", "_nfe4", 256, 0,
     "block_size=16 被压到每块 1-4 步，属其设计区间外"),
    ("baseline", "BD3-LM 限步 128（每块 2 步）",
     "benchgen_bd3lm_owt2", "_nfe2", 128, 0, "同上；限步曲线非单调"),
    ("baseline", "BD3-LM 限步 64（每块 1 步）",
     "benchgen_bd3lm_owt2", "_nfe1", 64, 0, "同上"),
    ("baseline", "AR（GPT-2 架构）", "benchgen_ar_owt2", "_final", 1024, 0, ""),
    ("baseline", "MDLM 满预算", "benchgen_mdlm_owt2", "_final", 1024, 0, ""),
    ("baseline", "MDLM 限步 64", "benchgen_mdlm_owt2", "_mdnfe64", 64, 0, ""),
    ("baseline", "MDLM 限步 22", "benchgen_mdlm_owt2", "_mdnfe22", 22, 0, ""),

    # ---- degenerate baselines: ROUGE not comparable ---------------------
    ("degenerate", "SSD-LM T=10", "benchgen_ssdlm_owt2", "_S10", 410, 0,
     "退化：高频功能词汤，distinct-2 0.886、prompt bigram 复制率仅 6.8%，MAUVE 地板"),
    ("degenerate", "隐扩散 CADENCE-LDM 64 步",
     "benchgen_ldiff_owt2_pqsh", "_D64", 64, 0, "退化：词沙拉 + token 复读"),
    ("degenerate", "CMLM T=64", "benchgen_cmlm_owt2", "_T64", 64, 0, "退化"),
    ("degenerate", "CMLM T=22", "benchgen_cmlm_owt2", "_T22", 22, 0, "退化"),
    ("degenerate", "CMLM T=10", "benchgen_cmlm_owt2", "_T10", 10, 0, "退化"),
    ("degenerate", "CMLM T=4", "benchgen_cmlm_owt2", "_T4", 4, 0, "退化"),
    ("degenerate", "TextLDM 架构复现 w=7, 50 步",
     "benchgen_textldm_dit_owt2", "_finalTLDM", 50, 0,
     "退化：高熵词碎片；DiT 精确 2.0002B，VAE 39.3B 单独披露"),
]

GROUP_TITLE = {
    "cadence": "CADENCE 尝试（按机制递进，全部严格 2B）",
    "axis": "三个尺度内解码轴的单变量对比（同父、同 7630 步、depth 全冻结）",
    "hmar": "P1：HMAR §4.3 尺度重加权",
    "baseline": "流畅 baseline（可直接对比）",
    "degenerate": "退化 baseline（ROUGE 不可与流畅系统并读）",
}

# Selection-set sweeps (n=250, disjoint from test). These are DECODE settings on
# a fixed checkpoint, so they answer "which setting", not "which model".
# They are reported separately from test because n=250 MAUVE has a measured
# bootstrap sd of 3.2-5.3 -- see the caveat printed under each sweep.
SEL_SWEEPS = [
    ("混合覆盖扫描（`b2s2dm`，粗带 seg + 细带 2D，全尺度覆盖）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2dm",
     [("seg粗K2 + 2D细C2K1（44）", "_m44"),
      ("seg粗K4 + 2D细C2K2（88，与主线 iso-NFE）", "_m88"),
      ("seg粗K4 + 2D细C4K2（112）", "_m112"),
      ("对照 seg:all:4 同臂（88）", "_msegall")],
     "覆盖固定、NFE 匹配后 2D 仍输给纯段轴：m88 15.70 vs 同臂对照 19.09（wiki），"
     "WS 同方向（17.84 vs 31.11）——两集一致，P0 方向就此彻底关闭。"),
    ("2D MaskGIT 扫描（`b2s2d`，lrseg:8,9,10:C:K，仅细尺度）",
     "benchgen_planner_prefix_owt2_pqsh_b2s2d",
     [("C=2 K=4（48，mentor 组1：段原样/位置粗）", "_2dc2k4"),
      ("C=8 K=1（48，组2：位置原粒度/段粗）", "_2dc8k1"),
      ("C=4 K=2（48，组3：双粗化）", "_2dc4k2"),
      ("C=2 K=2（24）", "_2dc2k2"),
      ("C=4 K=1（24）", "_2dc4k1"),
      ("锚 C=1 K=4 = 纯段轴（24）", "_2dc1k4")],
     "同 NFE 档内 2D 不优于纯段轴锚（两 sel 集对 C×K 排序反相关）；且所有"
     "细尺度-only 配置都远低于全覆盖 seg:all:4 的 22.89 —— 见下面的覆盖曲线。"),
    ("采样器尺度覆盖曲线（`b2sg` 同 checkpoint，seg K=4，只改尺度集）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg",
     [("{8,9,10} 细尺度-only（24）", "_segfine"),
      ("{0..7} 粗尺度-only（64）", "_sgcoarse"),
      ("{0..9} 全部除 q1024（80）", "_sgno1024"),
      ("{0..10} 全尺度（88，主线）", "_sgseg"),
      ("无采样器（0）", "_sgplain")],
     "**没有任何真子集接近全覆盖**：最好的 {0..9}@80 只到 15.62，而全覆盖@88 "
     "= 22.89；补上 q1024 那最后 8 次前向带来最大单跳 +7.3。为压 NFE 而砍尺度"
     "覆盖是亏本买卖 —— 这是 2D 扫描教给我们的真正结论。"),
    ("段轴 K 曲线（`b2sg`，唯一变量 = 每尺度承诺轮数）",
     "benchgen_planner_prefix_owt2_pqsh_b2sg",
     [("段并行（K=1 等价，走 plain 读出）", "_sgplain"),
      ("段 MaskGIT K=2", "_sgk2"),
      ("段 MaskGIT K=3", "_sgk3"),
      ("段 MaskGIT K=4 = S（完全顺序化）", "_sgseg")],
     "K 曲线单调，且在 K=S=4 有大跳跃 —— 段轴要的是完全顺序化的极限。"
     "注意 K=1 行走的是 plain 读出（`_sample_block`），不是 `seg:all:1`，"
     "所以它比真正的段并行对照多差一个变量。"),
    ("chunk 数扫描（`b2slr3`，K=1，唯一变量 = chunk 数 C）",
     "benchgen_planner_prefix_owt2_pqsh_b2slr3",
     [("lr C=2 K=1", "_sw2k1"), ("lr C=4 K=1", "_sw4k1"),
      ("lr C=8 K=1", "_sw8k1"), ("lr C=16 K=1", "_sw16k1"),
      ("lr C=16 K=4", "_lr3")],
     "**两个 sel 集把 C 排成几乎相反的顺序（Spearman ρ = −0.90）**，5 格的 sd 是 5.10，"
     "而最大值 21.88 恰好落在 E[max of 5] = 20.90 上 —— 没有可测的 chunk 数效应。"),
    ("位置轴对照（`b2spf`，纯 sampler_pos + depth 冻结）",
     "benchgen_planner_prefix_owt2_pqsh_b2spf",
     [("pos K=8", "_posf")],
     "chunk 数扫描的匹配对照。它在 wiki 的 R1/R2/RL/BERT/distinct-2 五项上"
     "胜过全部五个 lr 格。"),
]


def load(results: Path, run_dir: str, tag: str, bench: str):
    p = results / run_dir / f"gens_{bench}{tag}.metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"r1": d["rouge1"] * 100, "r2": d["rouge2"] * 100,
            "mauve": d["mauve"] * 100, "n": d.get("n"),
            "distinct2": d.get("distinct2"), "bertscore": d.get("bertscore_f1"),
            "path": str(p.relative_to(results.parent))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out_md", default="docs/reports/MASTER_TABLES.md")
    ap.add_argument("--out_csv", default="results/master_table.csv")
    args = ap.parse_args()
    results = Path(args.results)

    flat, missing = [], []
    for group, label, run_dir, tag, nfe_b, nfe_s, note in ROWS:
        for b in BENCHES:
            m = load(results, run_dir, tag, b)
            if m is None:
                missing.append(f"{run_dir}/gens_{b}{tag}.metrics.json")
                continue
            flat.append({"group": group, "label": label, "run_dir": run_dir,
                         "tag": tag, "benchmark": b, "nfe_backbone": nfe_b,
                         "nfe_sampler": nfe_s, "note": note, **m})

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    idx = {(r["label"], r["benchmark"]): r for r in flat}
    out = ["# 主表（脚本生成，勿手改）",
           "",
           "由 `tools/build_master_table.py` 从 `results/benchgen_*/` 的原始 "
           "`*.metrics.json` 重建。每格 = R1/R2/MAUVE ×100，n=1000（test 一枪）。",
           "除标注外全部严格 2B（7630×256×1024 梯度 token）、同数据、同 GPT-2 BPE、",
           "同 12L×768 主干。NFE = 每生成 1024 token 的 backbone 前向次数（CFG 双分支计入）。",
           ""]
    for group in ["cadence", "axis", "hmar", "baseline", "degenerate"]:
        rows = [r for r in ROWS if r[0] == group]
        if not any((r[1], b) in idx for r in rows for b in BENCHES):
            continue
        out += [f"## {GROUP_TITLE[group]}", "",
                "| 配置 | NFE | " + " | ".join(BENCH_LABEL[b] for b in BENCHES) + " |",
                "|---|---|" + "---|" * len(BENCHES)]
        for _, label, _, _, nfe_b, nfe_s, _ in rows:
            cells = []
            for b in BENCHES:
                r = idx.get((label, b))
                cells.append(f"{r['r1']:.2f}/{r['r2']:.2f}/{r['mauve']:.2f}"
                             if r else "—")
            nfe = f"{nfe_b}" + (f" (+{nfe_s})" if nfe_s else "")
            out.append(f"| {label} | {nfe} | " + " | ".join(cells) + " |")
        notes = [(lb, nt) for _, lb, _, _, _, _, nt in rows if nt]
        if notes:
            out += [""] + [f"- **{lb}**：{nt}" for lb, nt in notes]
        out.append("")

    # ---- selection-set sweeps: decode settings on a fixed checkpoint -----
    out += ["---", "",
            "# 选择集扫描（sel，n=250，与 test 不相交）", "",
            "这些是**固定 checkpoint 上的解码设置**扫描，回答的是"
            "「哪个设置」而不是「哪个模型」。**n=250 上 MAUVE 的 bootstrap "
            "标准差实测是 3.2~5.3 分**，所以单看一格的高低没有意义 —— "
            "每张表下面的那句话才是结论。", ""]
    for title, run_dir, cells, caveat in SEL_SWEEPS:
        out += [f"## {title}", "",
                "| 设置 | sel_wikipedia R1/R2/MAUVE | sel_wikisource R1/R2/MAUVE |",
                "|---|---|---|"]
        for label, tag in cells:
            got = []
            for b in ("sel_wikipedia", "sel_wikisource"):
                m = load(results, run_dir, tag, b)
                got.append(f"{m['r1']:.2f}/{m['r2']:.2f}/{m['mauve']:.2f}"
                           if m else "—")
                if m:
                    flat.append({"group": "sel", "label": f"{title} :: {label}",
                                 "run_dir": run_dir, "tag": tag, "benchmark": b,
                                 "nfe_backbone": 22, "nfe_sampler": 0,
                                 "note": "sel n=250", **m})
            out.append(f"| {label} | " + " | ".join(got) + " |")
        out += ["", caveat, ""]

    # rewrite the CSV so it carries the sel rows too
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    Path(args.out_md).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out_md} and {args.out_csv}: {len(flat)} cells, "
          f"{len({r['label'] for r in flat})} arms")
    if missing:
        # loud, not silent: a missing file means the table has a hole
        print(f"\nMISSING {len(missing)} files (rows rendered as '—'):")
        for m in sorted(missing):
            print(f"  {m}")


if __name__ == "__main__":
    main()
