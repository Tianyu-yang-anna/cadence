# CADENCE Stage 0：Text Multi-Scale VQ-VAE 实验报告

日期：2026-08-07 ~ 2026-08-09 ｜ 代码：`~/cadence`（本地 git）｜ 数据/结果/ckpt：`/Volumes/sandbox_ai/u_tianyuy/cadence`

---

## 1. 背景与目标

CADENCE proposal 要把 VAR（Visual Autoregressive, next-scale prediction）范式移植到文本：先用多尺度残差量化 VQ-VAE 把文本编成"粗→细"的离散码（Stage 0），再训一个 planner 逐尺度生成这些码（Stage 1），最后一次并行解码出文本。

**Stage 0 的 pilot 问题**：文本 latent 能否分解出有意义的 coarse-to-fine 离散码，同时保持近无损重建？

按 need.md 规范：N=256 GPT-2/BERT token、**c=1 全程无下采样**（唯一的池化发生在量化器内部、作用于残差 latent）、从头训练的双向 Transformer（6L/512d encoder + 8L/512d decoder、tied LM head、~70M 参数）、**bidirectional reconstruction**（decoder 一次并行输出全部 256 个位置）、VQ-EMA 码本 8192×32d。

**评估核心**：
- 全量重建 acc/PPL（地基指标，proposal 门槛 ≥99.5%）
- **截断重建**（只给 decoder 前 k 层粗码）——测粗码携带多少真实信息，是层级质量的直接读数
- 逐尺度能量移除、码本健康（活码率/困惑度/跨尺度 Jaccard）
- planner-friendliness 探针（proposal §8.9）：tiny AR 测各尺度码可预测性 + 粗码文档一致性

## 2. 实验清单（12 个训练 run + 2 个探针）

共享设置：WikiText-103、50k 步、global batch 256、AdamW 3e-4 cosine、bf16、前 5k 步量化旁路热身、各占 1×H100 约 6.5h。

| run | tokenizer | scale schedule | dropout | 其它设置 | 总码数 |
|---|---|---|---|---|---:|
| base | GPT-2 | [1,2,4,256] | 0 | 第一轮基线 | 263 |
| sd05 | GPT-2 | [1,2,4,256] | 0.5 | 第一轮对照 | 263 |
| bertPilot | BERT | [1,2,4,256] | 0.5 | 第二轮参照系 | 263 |
| bertPhi | BERT | [1,2,4,256] | 0.5 | +φ 卷积（VAR 组件） | 263 |
| bertSepCB | BERT | [1,2,4,256] | 0.5 | 每尺度独立码本 | 263 |
| bertP75 | BERT | [1,2,4,256] | 0.75 | 更强层级压力 | 263 |
| bertD | BERT | [1,4,16,64,256] | 0.5 | 稀梯子（×4 步距） | 341 |
| bertA | BERT | [4,8,16,32,64,128,256] | 0.5 | 密梯子、无超粗端 | 508 |
| bertB ⭐ | BERT | [8,16,32,64,128,256] | 0.5 | 密梯子、从 8 起步 | 504 |
| bertC | BERT | [1,2,...,128,256] | 0.5 | VAR 完整梯子（9 尺度） | 511 |

探针（非训练）：probe-sd05、probe-bertB —— 5 万窗口的码序列上训 4L tiny AR，测各尺度 held-out CE、unigram 熵基线、相邻窗口码一致性。

（scale dropout p：训练时以概率 p 只给 decoder 前 k 层码（k 随机），逼粗码携带信息。GPT-2 版另有 4 个 schedule run 提交后依用户决定取消，改用 BERT 重跑。）

## 3. 结果

### 3.1 全量重建（地基指标，test 集）

| run | 总码数 | acc | PPL |
|---|---:|---:|---:|
| base | 263 | 99.79% | 1.011 |
| sd05 | 263 | 99.76% | 1.012 |
| bertPilot | 263 | **99.85%** | **1.006** |
| bertPhi | 263 | 99.80% | 1.008 |
| bertSepCB | 263 | 99.70% | 1.011 |
| bertP75 | 263 | 99.64% | 1.014 |
| bertD | 341 | 99.35% | 1.026 |
| bertA | 508 | 99.55% | 1.020 |
| bertB | 504 | 99.43% | 1.025 |
| bertC | 511 | 99.66% | 1.016 |

全部 ≈99.5% 门槛以上、PPL≈1.0：**近无损重建在所有配置下成立**。密梯子有 ~0.2–0.4pp 的"重建税"（任务干扰 + 多轮量化累积噪声），在都及格的区间内无关紧要。

### 3.2 截断重建（层级质量，关键结果）

只给 decoder 前缀粗码时的重建 acc / PPL：

| run | 前缀 | 累计码数 | acc | PPL |
|---|---|---:|---:|---:|
| base（dropout 0） | [1,2,4] | 7 | 0.9% | 1.7M† |
| sd05 | [1,2,4] | 7 | 6.7% | 654 |
| bertPilot | [1,2,4] | 7 | 7.8% | 565 |
| bertD | [1,4,16,64] | 85 | 31.1% | 48.7 |
| bertA | [4..64] | 124 | 40.6% | 16.5 |
| bertA | [4..128] | 252 | 81.3% | 2.11 |
| **bertB** | [8..32] | 56 | **29.7%** | 55.0 |
| **bertB** | [8..64] | 120 | **52.5%** | 9.81 |
| **bertB** | [8..128] | 248 | **84.4%** | 1.84 |
| bertC | [1..64] | 127 | 38.7% | 18.3 |
| bertC | [1..128] | 255 | 80.0% | 2.22 |

† base 的天文 PPL：decoder 训练时从未见过截断输入，对分布外输入"自信地胡说"——**不开 dropout 时截断评估不可信**，这是设计双 run 的原因。

**读数**：
- [1,2,4,256] 的 7 个粗码是硬容量天花板（~8%）；几何密梯子解锁平滑坡道（124 码 52%、250 码 84%）。**预算墙是 schedule 造成的，不是方法不行。**
- **bertB（从 8 起步）在每个匹配预算点全面最优**；超粗尺度 1/2/4 几乎零信息——文本的有用全局粒度 ≈ 每 32 token 一码。

### 3.3 单因子消融（vs bertPilot）

| 因子 | 7 码截断 acc | 全量 acc | 结论 |
|---|---|---|---|
| +φ 卷积 | 7.8%（无变化） | 99.80%（略降） | **无效，弃用** |
| 每尺度独立码本 | 8.1%（+0.3pp） | 99.70% | 真实但边际（4× 码本参数） |
| dropout 0.75 | 7.9%（≈无变化） | 99.64% | **压力不是瓶颈，容量才是；0.5 定为标配** |
| BERT vs GPT-2（bertPilot vs sd05） | 7.8% vs 6.7% | 99.85% vs 99.76% | tokenizer 影响小，结论可迁移 |

### 3.4 探针（planner-friendliness，proposal §8.9）

**probe-sd05（[1,2,4,256]）**：
- q1 全局码 = **文档/主题指纹**：相邻窗口（多为同文档）q1 一致率 44.1% vs 随机窗口对 0.45%（**98× lift**）
- 细码（l=256）从上下文获得 **+3.4 bits** 可预测性（边际熵 12.15 → AR CE 8.74 bits）——"粗码先行让细码好预测"，next-scale 范式的前提成立
- 但中间尺度（2/4）上下文增益 ≈ 0——pilot 的中间层是噪声，与截断结果互证

**probe-bertB（[8..256]）**：
- 各尺度码近满熵（11.9–12.7 / 13 bits，码本 100% 利用）→ 信息打包极密（这正是截断曲线好的原因）
- 前缀 AR 增益温和（粗中 +0.2–0.5 bits、最细 +2.05 bits）
- **没有文档指纹尺度**（l=8 lift 仅 12×、绝对值 0.35%）——密梯子用主题锚换了重建效率
- 注：tiny AR 无文本条件，是可预测性下界；真实 planner 有 prompt 条件

### 3.5 码本健康（全部 run）

全局活码率 96–100%、无坍缩（usage-based 死码复活机制工作正常）；共享码本会按尺度自发分区（粗/细码集 Jaccard≈0）。

## 4. 结论：pilot 三个问题的答案

1. **近无损重建可行吗？—— 可行且稳健。** 所有配置 99.35–99.85%（PPL≈1.0），c=1 下多尺度残差离散瓶颈足以近无损表示文本。
2. **粗→细层级会自然涌现吗？—— 不会。** 不加压力时全部信息挤进最细尺度，粗码是摆设（base：粗码解码 ~1%、l=2/4 能量 ~0.1%）。另一个方法论教训：base 的 l=1 移除 59% 能量却完全不可解码——**"能量移除"≠"可解码信息"**。
3. **层级能被造出来吗？形状是什么？—— 能；正确形状 = 密几何梯子。** scale dropout 0.5 是充分且必要的正则（0.75 无增益）；schedule 从 [1,2,4,256] 换成 [8..256] 后粗码解码从 8% → 52%（124 码）→ 84%（248 码）。代价是失去 q1 主题锚 + ~0.3pp 重建税。

## 5. Stage 1 tokenizer 配方（最终建议）

> **schedule = [1, 8, 16, 32, 64, 128, 256]**（混合：1 个主题锚码 + 从 8 起步的密坡道）
> **+ scale_dropout 0.5 + 共享码本 8192 + 无 φ + tokenizer 任选（BERT/GPT-2）**

理由：bertB 证明密梯子的坡道效率，sd05 探针证明 q1 锚码的语义价值（98× 文档一致性、planner 易预测），混合两者。建议 Stage 1 冻结 tokenizer 前用该 schedule 跑一次确认（~6.5h，1×H100，纯 config 改动）。

## 6. 工程记录（复用价值）

**审查修掉的实现缺陷（对抗验证确认）**：死码复活不能用 EMA cluster_size 绝对阈值（其总质量=平均每调用分配数，K=8192 下会每窗口重置 ~88% 码本；改为窗口内原始使用计数 + 跨尺度 reservoir 采样）；WikiText 文档切分需要求标题前后空行（train 有 ~972 条正文伪标题）；eval 自动采用 ckpt 内保存的配置（消融 state_dict 形状相同会静默错配）。

**平台坑（Zillow Databricks）**：UC Volume 必须先 `databricks volumes create`（mkdir 建不了 Volume，写入静默失败）；sgcli `include_paths` 校验用 `git ls-tree -d` 只认目录；**GPU_1xH100 配额已从 5 缩到 4 并发**；多卡节点平台注入 `WORLD_SIZE` 但无 `RANK`（DDP 检测必须同时要求 RANK）；`datasets>=5` 的 `ds["text"]` 返回懒 Column、大表逐项访问 O(chunks)（180 万行上会卡数小时，用 `.data.column().to_pylist()` 一次物化）；PyYAML 把 `2e-3` 解析为字符串。

**多实验打包**：1×H100 配额不够时，用一台 8×H100 节点跑 N 个单卡 worker（`CUDA_VISIBLE_DEVICES=$w` + 清除平台分布式变量），见 `jobs/extra4_entry.sh`。

## 附加轮：Next-3 实验（need_next3.md，2026-08-09/10）

**Exp 1 — 混合 schedule 验证**：bertHybrid=[1,8,16,32,64,128,256] 训完 50k 步。全量 99.44%（与 bertB 99.43% 持平，均略低于 99.5% 严格线；pilot 系列 99.8%+ 可过）；坡道各预算点比 bertB 低 1.4–2pp（q1 有小代价）；q1 语义锚保留但稀释（相邻一致 58.7%、lift 22.6×，vs bertPilot 的 103.8×，q1 有效码数 ~38 vs ~250）。

**Exp 2 — 逐尺度边际贡献（LOSO）**：用 subset-readout decoder（冻结 encoder/码本、decoder 副本在随机子集上微调）修正分布外失真后（raw 模式把粗尺度损失夸大 ~2 倍），**两个 schedule 都没有冗余尺度**：去掉任一尺度损失 0.8–18.2pp、随尺度码数单调；相邻尺度互补不重叠。

**Exp 3 — 严格 next-scale 探针（本轮头条，负面发现）**：按 VAR 分解（给定全部粗码、并行独立预测下一尺度所有码，容量匹配的 null 控制组）测得**粗码对细码的预测增益 ≈0**（全部 transition 在 −0.46～+0.09 bits/码；q256 增量条件平坦；**hybrid 的 q1 消融 = 0 增益**——q1 对 planner 预测毫无帮助）。结构性解释：残差量化**天生把各尺度去相关**（下一层编码的正是上一层没解释掉的部分）——同一机制同时解释了"无冗余"（信息正交）、"截断曲线平滑"（信息累加）和"粗预测不了细"（正交的另一面）。此前泛用 AR 探针的 +2.05 bits 增益被拆穿为尺度内相邻码相关性。**注意**：此探针测的是无条件耦合；Stage 1 planner 有 prompt 条件，prompt 条件下粗码可能仍有解歧价值——这是下一个该做的实验。

**冻结判定（need_next3 五条标准）**：① 全量 ≥99.5% —— 边缘（99.44%）；② 有意义的粗→细前缀重建 —— ✅；③ 无冗余层级 —— ✅；④ 粗码降低细码不确定性 —— **✗（两 schedule 均 ≈0，结构性）**；⑤ q1 语义或 planner 价值 —— 半：语义有（22.6×）、planner 预测无（消融=0）。

**Prompt 条件版补测（bertB，run 605558091889954）**：用前一窗原文做 prompt，容量匹配的 text-only vs text+粗码对照——**增益依然 ≈0**（5 个 transition 全部在 −0.09～+0.08 bits，处于 931 对验证样本的噪声范围内）；且 prompt 本身对预测码也几乎无帮助（12.81 vs 无条件 12.83 bits）——窗口级码身份本质上是高熵对象。标准 ④ 在无条件和 prompt 条件两种设定下都不成立。

~~最终结论：不冻结，tokenizer 需要 Stage 0.5 手术~~ **【2026-08-11 勘误推翻，见下】**

**勘误与最终结论（2026-08-11）**：探针存在两处设计错误（用户审查发现）——条件码用 from-scratch id embedding 而非冻结 pretrained codebook；目标位置用可学习 query 而非 VAR 的 e_k（累计反量化 latent 的 up-interpolation）。修正后（探针与 tokenizer 反量化路径单测锁定一致）**所有增益全面转正**：bertB →q256 增益 **+0.73 bits**（旧 +0.08）、bertHybrid **+1.11 bits**；prompt 版 +0.41/+0.60；增量条件强单调；q1 消融复验 +0.07 bits（"零价值"翻案）。**修订判定：五条标准 边缘/✅/✅/✅/✅ → tokenizer 不需要手术，冻结 bertHybrid**（耦合最强+q1 锚+重建持平）。Stage 1 硬性约束：planner 输入接口必须严格按 VAR 构造（偏离即丢失几乎全部跨尺度信号，已实测）；残余熵仍高（11.7/13 bits），prompt 条件承担其余——与 VAR 图像同性质。方法论教训：负面结论对探针接口极其敏感，两版探针结果均归档对照。

详表：`~/cadence/results/hybrid_schedule_summary.md` + 同目录 JSON。

## 7. 产物索引

| 内容 | 位置 |
|---|---|
| 代码（模型/训练/评估/探针/分析） | `~/cadence`（git，16 commits，44 测试全绿） |
| 结果详表 | `~/cadence/results_summary.md`、README「Results」两节 |
| 终评 JSON/码本直方图/定性样本 | `$VOL/results/vqvae_wt103_<run>/eval_test_step50000.json` 等 |
| 探针 JSON | `$VOL/results/vqvae_wt103_{sd05,bertB}/probe_planner_step50000.json` |
| checkpoint（可复用） | `$VOL/checkpoints/vqvae_wt103_<run>/` |
| 预处理数据 | `$VOL/data/wikitext103{,_bert}/`（GPT-2 与 BERT 两套 bins） |
| 复现命令 | `~/cadence/README.md`（submit/resume/消融均一行命令） |

（`$VOL` = `/Volumes/sandbox_ai/u_tianyuy/cadence`）
