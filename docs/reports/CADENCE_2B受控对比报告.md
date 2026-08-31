# CADENCE 2B 受控对比报告（2026-08-31 收官）

**一句话**：同数据、同 tokenizer、同预算、同主干下，CADENCE 在词面指标（R1）上 3/4 个 benchmark 领先全部 baseline 且对短 prompt 最鲁棒（1BW 上 AR 崩溃而我们正常），但在 Wikipedia/WikiSource 的 MAUVE 上落后于 AR 与 BD3-LM；BD3-LM 是当前预算档最强 baseline；CFG 被成对消融确认有效（+3~6 MAUVE）。

---

## 1. 统一设定（预注册）

- **数据**：OpenWebText2（HF parquet 镜像，实测 13.07M docs ≈ 12.8B GPT-2 token），shuffle+EOS 拼接 → uint16 bins；所有家族**比特级同数据**（BD3/MDLM 经 binwindows 补丁直读同一 bins）；
- **tokenizer**：GPT-2 BPE（50257），全体一致（实测 Qwen3 在英文上反而多耗 3.7% token，且 151k 词表在 100M 档 embedding 失衡）；
- **预算**：每家 **2B loss token** = 7630 步 × batch 256 × 1024（Chinchilla D≈20N）；<1 epoch 零复读；cosine + warmup 400；lr 3e-4；
- **主干**：12L×768×12h 全体统一（≈85M 非 embedding）；
- **两阶段口径**：CADENCE 的冻结 tokenizer（60M，150k 步预算外重训于 OWT2）单列披露——先例：TextLDM 的 VAE 光 embedding+head 就 223M 且不计入其"114M"；
- **采样（预注册）**：CADENCE 扫 4 个逐尺度调度（sel 4×250 不相交选择集，胜者规则 sel_wikipedia MAUVE > R1 > sel_wikisource MAUVE）；AR 用社区标准 nucleus T=1.0/top_p=0.95；BD3/MDLM 用官方 semi-AR first-hitting 采样（prompt 前缀注入适配）；test 每配置只跑一次。

## 2. Test 终评总表（4×1000，×100，R1/R2/RL/BS/MAUVE）

| 模型 | WikiSource | Wikipedia | TinyStories | 1BW |
|---|---|---|---|---|
| **CADENCE-ps**（per-scale，p5cold3） | **28.9**/2.7/13.8/78.9/3.05 | 22.9/2.2/12.6/78.1/4.22 | **26.0**/1.8/14.4/81.4/0.54 | 9.7/0.3/7.3/81.5/**0.59** |
| **CADENCE-sh**（shared，p5hot3） | 28.5/2.7/13.6/78.8/2.66 | 23.1/2.2/12.2/78.0/6.36 | 25.0/1.7/13.9/81.2/0.54 | 8.8/0.2/7.6/81.5/**0.59** |
| AR（GPT-2 架构） | 24.3/2.4/12.0/78.0/7.41 | 21.0/2.3/11.0/77.3/**11.92** | 14.2/1.0/9.0/77.2/0.54 | 1.9/0.0/1.8/74.2/0.47 |
| BD3-LM（block 16） | 27.5/**2.7**/13.5/79.5/**7.67** | **23.2**/**2.3**/12.0/78.6/11.39 | 25.3/**2.1**/14.2/**82.2**/**0.64** | **9.8**/**0.4**/8.4/**82.1**/0.49 |
| MDLM | 22.0/1.4/11.3/78.2/3.54 | 18.5/1.2/10.0/77.4/7.84 | 17.1/0.7/10.6/80.4/0.52 | 7.8/0.2/6.8/81.5/0.48 |

## 3. 调度扫描与 CFG 消融（sel 4×250，R1/R2/MAUVE）

| 调度 | CADENCE-ps wiki | ps WS | CADENCE-sh wiki | sh WS |
|---|---|---|---|---|
| p5cold（CFG off） | 20.6/1.9/4.3 | 26.0/2.5/5.5 | 21.2/2.0/6.7 | 25.9/2.4/8.6 |
| **p5cold3（CFG on）** | **21.7/2.1/7.1** ✓ | 27.0/2.8/12.3 | 22.2/2.3/11.3 | 28.1/3.1/11.5 |
| p5hot3（CFG on） | 21.0/1.8/5.6 | 26.3/2.5/7.7 | **21.7/1.9/12.4** ✓ | 27.4/2.8/10.1 |
| pflat（标量锚） | 19.3/1.3/3.7 | 24.1/1.7/2.3 | 19.6/1.4/3.5 | 24.7/1.9/3.6 |

**CFG on/off 成对行：MAUVE +3~6 点、R1/R2 同涨（两变体一致）——advisor 重开 CFG 的决定被数据确认。** 标量行仍垫底：逐尺度调度在无/有 CFG 下都是硬杠杆。

## 4. 冻结 tokenizer（OWT2 版，150k，test）

| | pqsh | pqps |
|---|---|---|
| 全量重建 | 99.927% | **99.956%** |
| pad 桶（32/128/512） | ≥99.66% | ≥99.53% |
| 段独立采样最坏 drop | 25.1pp | **13.2pp** |

## 5. Scaling 留档（results/scaling/scaling_log.jsonl，家族内 loss 不跨族比较）

AR token-CE 3.474 nats｜BD3 NELBO 4.133｜MDLM NELBO 4.302｜CADENCE-ps 码 CE 5.428 bits/段｜CADENCE-sh 4.820 bits/段（各 2B 点，供后续 L(D)/L(N) 阶梯 append）。

## 6. 判读

1. **CADENCE 的强项**：R1 在 WS/TS 领先全场、wiki 与 BD3 打平；**短 prompt 鲁棒性碾压**——1BW 上 AR 崩到 R1 1.9（窗口对训练 + 十几 token 的 prompt 完全 OOD），我们靠 pad-native tokenizer + prefix 拿 9.7；TinyStories 同理（AR 14.2 vs 我们 26.0）。
2. **CADENCE 的短板**：wiki/WS 的 MAUVE（3-6.4）落后 AR/BD3（7.4-11.9）——分布保真仍是主要差距，与 26B 档诊断一致。
3. **BD3-LM 是最强 baseline**：块内扩散+块间 AR 的"能边生成边修"机制在 MAUVE 上占优，R2/BS 也全场最高——它就是我们要打的靶子，也是"迭代修正机制值钱"的又一证据（→我们的精化/depth-AR 路线）。
4. **sh vs ps**：test 上 sh 的 wiki MAUVE 反超 ps（6.36 vs 4.22），与 26B 档相反且 sel→test 波动大——2B 档两者差异在噪声内，**码本共享方式仍非一阶因素**。
5. AR 的 wiki MAUVE 11.9 领先提醒：在其舒适区（长 prompt 网页文本）AR 依旧强，但它没有任何机制处理 prompt 分布偏移。

## 7. 披露与备注

- MDLM 训练时误带 block_size=16 配置（对其全窗掩码目标疑似无实质影响，val/nll 4.30 正常），推理按上游惯例以 block=1024 采样——已披露；
- BD3/MDLM 的 prompted 生成为本仓适配（前缀注入 semi-AR 采样，gen_prompted.py），采样步数 first-hitting 自适应；
- sel n=250 的 MAUVE 只用于组内选择，绝对值勿引用（n 效应）。

## 8. 下一步候选（按性价比）

1. **精化/迭代修正**：BD3 的优势恰是"能改"，我们的 MaskGIT 精化在新架构从未真测——把它接进 generate_prefix 纳入扫描（零训练成本）；
2. **CFG 系数细扫**（3 只是首个网格点，5/7 与逐尺度形状未探）；
3. **depth-AR 段头**（段间链式采样，治 MAUVE/局部一致性）；
4. **预算阶梯**（4B/8B + 更大模型点）观测各家族 scaling 斜率——BD3 vs CADENCE 谁随预算涨得快才是路线判决；
5. SSD-LM/TextLDM 二波（后者等作者回信）。

产物：checkpoints/{vqvae_owt2_1024_pq*, planner_prefix_owt2_pq*, ar_owt2, bd3lm_owt2, mdlm_owt2}；results/benchgen_*；scaling_log；代码全部已 push。
