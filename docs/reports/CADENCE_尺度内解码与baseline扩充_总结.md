# CADENCE：尺度内解码顺序三实验 + baseline 扩充 —— 总结（2026-09-02/03）

独立可读的汇总。逐章过程与原始数字见 `CADENCE_2B调优波报告.md` 终章五~九；
本文只给结论、证据链和可写进论文的主张。所有数字由脚本从 `results/` 的原始
`*.metrics.json` 重建。

---

## 0. 一句话

在严格同预算（2B 梯度 token、同数据、同 12L×768 主干）下，**段轴置信度序解码
（segment-wise MaskGIT）成为新主行：Wikipedia 的 R1 与 MAUVE 同时超过满预算的
BD3-LM，而 backbone 前向次数只有 22 次，是 BD3 的 1/46.5**；位置轴的两种顺序
机制（MaskGIT、chunk=1 AR）都无效，且已排除"采样头容量不足"这一解释。

---

## 1. 三个实验的答案

采样有两个正交轴：**位置轴**（一个尺度内的 l 个位置）× **段轴**（一个位置内的
S=4 个 PQ 段）。三个实验分别测这两个轴上的顺序机制。

| 实验 | 机制 | 结论 |
|---|---|---|
| EXP-1 | 段轴 MaskGIT（置信度序） | **有效，成为新主行** |
| EXP-2 | 位置轴 MaskGIT（置信度序） | 无效，**且与采样头容量无关** |
| EXP-3 | 位置轴 chunk=1 AR（严格左到右） | 无效，三者中最差 |

### 1.1 EXP-1：段轴是唯一有效的尺度内顺序机制

同权重（`_b2sg`）、唯一变量是解码顺序：

| 解码 | sel wiki R1/R2/MAUVE | sel WS MAUVE |
|---|---|---|
| 段并行 | 21.89/2.05/7.16 | 11.41 |
| 段 MaskGIT K=2 | 21.93/2.23/15.58 | 10.95 |
| 段 MaskGIT K=3 | 21.99/2.31/16.25 | 9.11 |
| **段 MaskGIT K=4 = S** | 22.33/2.33/**22.89** | **20.42** |

净效应 **+15.73 sel wiki MAUVE**。K 曲线单调，且在 **K=S=4（每轮只承诺一个段，
完全顺序化）** 有一个大跳跃——段轴要的是完全顺序化的极限，不是"少数几轮迭代"，
所以帕累托上没有更便宜的甜点。代价是**零额外 backbone 前向**（22 次不变，仅加
88 次 2 层浅前向 = +6.5% FLOPs）。

### 1.2 EXP-2：采样头容量不是原因

同权重（`_b2sp`）、同 NFE=64、同 K=8、同尺度 {8,9,10}，**唯一变量是采样头**：

| 采样头 | sel wiki MAUVE | sel WS MAUVE |
|---|---|---|
| 新 STAR 式采样器（4.16M） | 13.14 | 10.22 |
| 旧 linear+gate（25,345 参数） | **15.79** | **15.93** |

**把采样头放大 164 倍，位置轴反而更差**（wiki −2.65、WS −5.71），R1/R2 打平。
这是带匹配对照的干净负面结果，直接否证了"位置轴机制没效果是因为采样头太弱"。

### 1.3 EXP-3：位置轴上越顺序化越差

test 完整排序（R1/R2 三者几乎无差，23.2~23.4 / 2.3~2.5）：

**段轴 11.96 > 位置轴 MaskGIT 11.11 > chunk=1 位置 AR 8.23**（wiki MAUVE）

工程侧记：完整 `ar:8,9,10` 在本 harness 不可行——KV-cache 已把采样器 token-
forward 降 768×，但每位置的 aten 调度数几乎未减，GPU 上是 launch-bound，250 行
75.6 分钟未完成即被平台终止（两次复现）。注册行为单尺度 `ar:8`。这是工程限制，
不是机制结论，如实分开陈述。

---

## 2. 机制：为什么段轴赢、位置轴怎么都不行

两条**零训练、模块无关**的独立证据。

**(a) 耦合探针**（只测冻结 tokenizer + 解码器：把某尺度 p% 的位置在 batch 内
roll 打乱、保持边缘分布，看重建 token 准确率掉多少）：

| 尺度 | p=5% | p=10% | p=20% | p=40% |
|---|---|---|---|---|
| 8 (l=256) | 0.34pp | 0.73pp | 1.62pp | 4.42pp |
| 9 (l=512) | 0.50pp | 1.24pp | 2.92pp | 7.91pp |
| 10 (l=1024) | 0.23pp | 0.52pp | 1.32pp | 4.33pp |

对照本项目早先的**段间**耦合探针：最坏 **25.1pp**（正是它当初促成 depth-AR）。
**段间耦合比位置间耦合强一个数量级。**

**(b) masked vs next-scale 预测准确率**（reveal=50%，对标 HMAR App. D.3 的
5% → 65%）：最细尺度 0.277 → 0.350（+7.3pp），尺度 9 +4.5pp，尺度 8 +1.4pp，
粗尺度为负。**HMAR 在图像域报告的巨大 masked-gain 在文本 + 残差 PQ 上不成立。**
（保留：masked 一列经 `_b2sq2` 的 visible 通路测得，该 ckpt 的 `visible_gate`
仅 0.055，通路训练很弱；耦合探针不经过 planner，无此保留。）

---

## 3. 效率：从"更便宜"升级为帕累托支配

按 advisor 的要求把 baseline 的推理步数压到与我们相同（BD3 用
`sampling.first_hitting=False` + `num_steps`，MDLM 用 T）：

| 模型 | NFE | wiki R1 | wiki MAUVE |
|---|---|---|---|
| BD3-LM 满预算 | 1024 | 23.2 | 11.39 |
| BD3-LM 限步 | 256 | 18.2 | 0.57 |
| BD3-LM 限步 | 128 | 11.7 | 0.46 |
| **BD3-LM 限步** | **64** | **16.1** | **0.58** |
| MDLM 限步 | 64 | 18.5 | 5.63 |
| **CADENCE `_finalSEG`** | **22** | **23.4** | **11.96** |

**同样 64 次前向，BD3 的 wiki MAUVE 是 0.58，我们是 11.37~11.96**；而我们 22 次
前向就达到 BD3 满 1024 次的质量。**任何预算点上都没有 baseline 落在我们右上方。**

必须披露：BD3 训练时 block_size=16，压到每块 1~4 步是在其设计区间之外；限步曲线
**非单调**（NFE 128 比 NFE 64 更差），说明该区间本身不稳定。

另外一条与质量结果无关、恒成立的收获：STAR 式采样器把精化从"每轮重跑 12 层 ×
3071"改成"主干每尺度只算一次 + 2 层浅网络迭代"，**注册行的 backbone forward
从 64 降回 22，对 BD3 的优势从 16.0× 变成 46.5×**，且缓存是逐比特精确的
（`_assemble` 在不用输入侧通路时是纯函数、`maps` 在循环内不变）。

---

## 4. baseline 扩充：三个新家族在 2B 下全部退化

同数据、同 bins、同 12L×768 主干、同 2B 预算，自建最小实现：
**CMLM/Mask-Predict**（MaskGIT 的文本直系祖先）、**SSD-LM**（半自回归 simplex
扩散）、**CADENCE-LDM**（在我们自己的冻结 PQ 隐空间做高斯扩散）。

| 模型 | wiki R1/R2/MAUVE | distinct-2 | prompt 复制率 | 症状 |
|---|---|---|---|---|
| SSD-LM T=10 | **24.2/3.2**/0.50 | 0.886 | 6.8% | 高频功能词汤 |
| CADENCE-LDM 64 步 | 20.2/1.5/0.59 | 0.864 | 1.9% | 词沙拉 + token 复读 |
| CMLM T=64 | 11.6/1.7/0.88 | 0.353 | 14.4% | 严重退化 |

**方法论发现（对论文有独立价值）**：SSD-LM 拿到**全表最高的 ROUGE-1**（四个
benchmark 全部第一，1BW 14.4 远超 BD3 的 9.8），而它生成的是语法破碎的功能词汤、
MAUVE 全在 0.5 地板，且**没有抄 prompt**（bigram 复制率仅 6.8%）。高 R1 完全来自
"the/of/and/a"这类高频词与参考文本的单元重合。

结论：①这三家在 2B 下不构成有竞争力的 baseline，应作"预算下限示范"报告；
②**可比的流畅系统只有 AR / MDLM / BD3-LM / CADENCE 四家**，主表以它们为准；
③**只报 ROUGE 的对比不安全**，MAUVE 正确地把三个退化家族全判到地板。

---

## 5. 论文可用的三条主张

1. **同预算对 BD3-LM 的双超 + 46.5× 便宜**：`_finalSEG` 在 Wikipedia 的 R1
   （23.39 vs 23.2）与 MAUVE（11.96 vs 11.39）同时超过满预算 BD3-LM，R1 在
   四个 benchmark 里三个全场第一，backbone 前向 22 vs 1024。
2. **同 NFE 下 baseline 坍塌 → 帕累托支配**：把 BD3/MDLM 压到我们的前向预算，
   wiki MAUVE 掉到 0.46~5.63；不存在落在我们右上方的 baseline 配置。
3. **段轴 vs 位置轴的机制二分**：由三条独立证据支撑——匹配对照的采样头实验
   （164× 容量反而更差）、模块无关的耦合探针（25.1pp vs 1.3-2.9pp，差一个
   数量级）、以及 HMAR 的 masked-gain 在我们这个 regime 不迁移。

---

## 6. 必须披露的保留

- **sel→test 的 MAUVE 历史上反相关过**（三个注册臂 Spearman = −1），且
  `eval_generation.py` 的 MAUVE `num_buckets='auto'` = n/10，sel 用 ~25 簇、
  test 用 ~100 簇——sel 与 test 的 MAUVE 不可放进同一句话。
- **BD3 限步在其设计区间之外**，且限步曲线非单调。
- **WikiSource 的 MAUVE 仍落后**（`_finalSEG` 5.10 vs BD3 7.67）。
- **1BW 的 R1 仍差约 1 分**（9.28 vs 9.8）——窗口规划范式对单句续写的固有税。
- **退化 baseline 的 ROUGE 不可与流畅系统并读**（见 §4）。
- **两阶段口径**：CADENCE 的冻结 tokenizer（60M，150k 步 = 39.3B token）在 2B
  预算之外，单独披露；CADENCE 的 planner 总参 88M 看似小于 baseline 的 124M，
  是因为它不带 50257 词嵌入/输出头——主干层面（85M 非嵌入）全家族一致。
- **行分片改造后**，改动前注册的 test 行任何新跑都不可逐比特复现（一个采样流不能
  既依赖前面所有行、又与分片无关）；已注册指标作为测量仍然有效。

---

## 7. 产物

**代码**（commit 8e9e131 起）：
- `models/sampling_transformer.py` —— STAR 式独立采样器（2L×384，4.16M）；输出
  768 维残差加到现有 per-scale head 的输入状态，零初始化即精确 no-op
- `models/prefix_planner.py` —— 三个解码模式 `pos` / `seg` / `ar`；位置轴走
  depth-AR 链、段轴 depth 冻结（两轴正交）；残差按尺度门控使训练读出 == 推理读出
- `finetune_prefix_maskgit.py` —— `sampler_pos` / `sampler_seg` / `sampler_causal`
  / `sampler_mix` 四种掩码模式
- `tools/diagnose_intra_scale.py` + `jobs/diag_entry.sh` —— 零训练机制诊断
- `models/cmlm_baseline.py` / `models/ssdlm.py` / `models/latent_diffusion.py`
  / `models/textldm_{vae,dit}.py` —— 四个新 baseline 家族
- `tests/test_sampling_transformer.py` —— 门禁测试，含"训练读出 == 推理读出"
  （在两个缺陷各自单独存在时都会失败）与 KV-cache 逐比特等价

**checkpoint**（Volume `checkpoints/`）：`planner_prefix_owt2_pqsh_b2sp`（位置轴臂）、
`_b2sg`（段轴臂）、`_b2sq1/_b2sq2/_b2mgd/_b2pl/_b2mg3/_b2ck/_b2nd`（此前各臂）、
`cmlm_owt2`、`ssdlm_owt2`、`ldiff_owt2_pqsh`、`textvae_owt2_g2`（训练中）

**结果**：`results/benchgen_*/` 全部 sel 与 test 的 JSONL + metrics.json；
`results/diagnostics/diag_sq2.json`

**在飞**：TextLDM 架构复现（commit b71357e）——Transformer VAE（1:1 token→latent、
dim 64、CE+KL+REPA）+ 流匹配 DiT，VAE 训练中，之后 DiT 精确 2B → sel → test。
定位是"**架构忠实、2B 预算的复现**"，不是"复现了 TextLDM 的结果"。
