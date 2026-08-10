# CADENCE Stage 0: Text Multi-Scale VQ-VAE

将 VAR（Visual Autoregressive Modeling, next-scale prediction）范式向文本迁移的第一阶段：训练一个**多尺度残差量化 VQ-VAE**，把 256-token 的文本窗口编码为从粗到细的离散码序列，并系统验证这套码是否满足下游 VAR-planner 的前提假设。

```
input (256 tokens)
   │  bidirectional Transformer encoder (6L, d=512)
   ▼
latent f ∈ R^{256×32}
   │  multi-scale residual VQ  (shared codebook 8192×32, VQ-EMA)
   │  for l in [8,16,32,64,128,256]:            ← scale schedule（可配置）
   │      pool(residual, l) → quantize → upsample → residual -= contrib
   ▼
codes: q_8 (8个) + q_16 + ... + q_256   (共 504 个离散码)
   │  bidirectional Transformer decoder (8L, d=512)，一次并行输出
   ▼
reconstructed 256 tokens
```

核心训练技巧：**scale dropout**（以概率 p 只把前 k 层码给 decoder），否则粗尺度不携带任何信息（见实验 1）。

**一页结论**：① 重建近无损（99.4–99.9% token acc）；② 层级不会自然涌现，dropout 可以逼出，密梯子 schedule 最优；③ 每层码无冗余；④ **但严格测量下粗码对细码的预测增益 ≈ 0**（残差量化天生使各层近独立）——纯残差码空间不满足 next-scale planner 的前提，tokenizer 未冻结，修改方向见文末。

---

## 环境与数据

```bash
uv venv && uv pip install --python .venv/bin/python -r requirements-local.txt
pytest tests/ -q                                          # 56 tests
# 数据：WikiText-103 → 文档感知分词 → uint16 bin（~3 分钟，勿在计算节点跑，见"坑"）
python data/prepare_wikitext.py --tokenizer bert-base-uncased --out <bin_dir>
```

训练单卡 H100 约 6.5h/run（~70M 参数、50k 步、batch 256）。我们用 Databricks serverless GPU（`jobs/` 下有全套提交脚本）；其它环境直接 `python train_vqvae.py --config ...` 即可，断点续训自动。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `models/multiscale_residual_vq.py` | 核心：多尺度残差量化循环（不变量有单测） |
| `models/vq_ema.py` | VQ-EMA + 基于使用计数的死码复活 |
| `models/{transformer,text_encoder,text_decoder,text_vqvae}.py` | 双向 Transformer 与整体组装（scale dropout / 截断 / 子集解码都在 `text_vqvae.py`） |
| `train_vqvae.py` / `eval_vqvae.py` | 训练；评估（全量+截断表+码本健康+样例）——所有实验共用 |
| `experiments/exp1_dropout_pilot/` | 实验 1：层级是否自然涌现（纯配置对照，README 含命令） |
| `experiments/exp2_schedule_sweep/` | 实验 2：schedule 扫描（纯配置对照） |
| `experiments/exp3_single_factors/` | 实验 3：单因子消融（纯配置对照） |
| `experiments/exp4_scale_redundancy/` | 实验 4 专属代码：readout decoder 微调 + 留一尺度/子集评估 |
| `experiments/exp5_next_scale_probe/` | 实验 5 专属代码：三个探针（泛用 AR / 严格 next-scale / +prompt） |
| `experiments/analyze_runs.py` | 跨 run 对比表生成 |
| `configs/` | GPT-2 / BERT / CPU-tiny 三个 yaml；一切消融走 `--set`，零代码改动 |
| `results/` | 全部 19 份结果 JSON + 冻结判定（`hybrid_schedule_summary.md`） |
| `docs/` | 实验需求文档（`specs/`）、两份完整中文报告（`reports/`）、详细英文日志 |
| `EXPERIMENTS.md` | 全部 run 的提交命令 + Run ID 索引 |

---

## 实验 1：层级会自然涌现吗？（base vs sd05）

**设置**：GPT-2 词表，schedule [1,2,4,256]，唯一变量 = scale dropout（0 vs 0.5）。
**代码**：`train_vqvae.py`；截断评估在 `utils/evaluation.py::evaluate`。
**结果**（test，50k 步；"前缀 k 码"= 只给 decoder 前 k 个粗码时的重建 acc）：

| run | dropout | 全量 acc / PPL | 前缀 7 码 acc | 前缀 7 码 PPL |
|---|---|---|---|---|
| base | 0 | 99.79% / 1.011 | 0.9% | 1.7M |
| sd05 | 0.5 | 99.76% / 1.012 | **6.7%** | 654 |

**分析**：不加压力时全部信息挤进最细层（base 的 l=2/4 只移除 0.1% 残差能量），且 base 的 decoder 从未见过截断输入、对其分布外（PPL 1.7M = 自信地胡说）→ **无 dropout 时层级不存在且截断评估不可信**。dropout 0.5 使粗码携带真实信息而全量重建只降 0.03pp。另一个方法论发现：base 的 l=1 移除 59% 能量却完全不可解码——能量指标 ≠ 信息指标。

## 实验 2：层级的最优形状（schedule 扫描）

**设置**：换 bert-base-uncased 词表（vocab 30522；对照实验证明词表影响 <0.1pp），全部 dropout 0.5，扫 5 种 schedule。
**结果**（test 截断 acc，按累计码数对齐比较）：

| schedule | ~60 码 | ~124 码 | ~250 码 | 全量 |
|---|---|---|---|---|
| [1,2,4,256] (bertPilot) | —（7 码封顶 7.8%） | — | — | **99.85%** |
| [1,4,16,64,256] (bertD) | — | 31.1% (85码) | — | 99.35% |
| [4,8,...,256] (bertA) | 17.8% | 40.6% | 81.3% | 99.55% |
| **[8,16,...,256] (bertB)** | **29.7%** | **52.5%** | **84.4%** | 99.43% |
| [1,2,4,...,256] (bertC) | 17.3% | 38.7% | 80.0% | 99.66% |

**分析**：粗码质量由 schedule 决定，**bertB 在所有预算点最优**。超粗尺度 (1/2/4) 是低效台阶——文本的有用全局粒度从"每 32 token 一码"（l=8）开始。密梯子有 ~0.2–0.4pp 的全量重建税（任务干扰 + 多轮量化累积噪声），在都 >99.4% 的区间内可忽略。

## 实验 3：单因子消融

**设置**：都在 bertPilot 上改一个因子。**结果**：

| 因子 | 截断变化 | 结论 |
|---|---|---|
| +φ conv（VAR 的上采样后卷积） | +0.05pp | 对文本无效，弃用 |
| 每尺度独立码本 | +0.3pp | 有效但边际，不值 4× 参数 |
| dropout 0.5→0.75 | +0.1pp | 瓶颈是粗码容量，不是训练压力 |
| GPT-2 → BERT 词表 | +1.1pp / 全量 +0.09pp | 词表选择不关键 |

## 实验 4：尺度冗余检验（leave-one-scale-out）

**动机**：前缀评估证明"加码有用"，但不能区分"每层独特"还是"相邻冗余"。
**方法**：藏掉任意一层码再重建。关键修正——decoder 只见过前缀输入，非前缀子集是分布外；先用 `finetune_subset_readout.py` 冻结 encoder/码本、微调 decoder 副本 2k 步，用它读数（实测 raw 模式把粗尺度损失夸大 ~2 倍）。
**结果**（readout 模式，去掉该层后的 acc 损失）：

| 去掉 | q1 | q8 | q16 | q32 | q64 | q128 | q256 |
|---|---|---|---|---|---|---|---|
| bertB | — | −3.3 | −4.4 | −6.7 | −9.3 | −16.3 | −17.2 pp |
| bertHybrid | −0.8 | −2.4 | −3.3 | −5.4 | −9.4 | −17.1 | −18.2 pp |

**分析**：**没有冗余尺度**——每层去掉都疼，损失随该层码数单调；邻居组合互补不重叠（[q64,q128,q256]=66.9% vs 抽掉 q128 后 33.1%）。单尺度独立解码普遍弱（8–23%），符合残差语义（每层编码"更粗层没解释掉的部分"）。

## 实验 5：粗码能预测细码吗？（Stage 1 的前提，本仓库最重要的实验）

**动机**：VAR planner 的假设 = "生成粗码后，细码变好猜"。前缀重建好 ≠ 这个假设成立。
**方法**（`probe_next_scale.py`）：按 VAR 的分解方式——给定**全部**更粗层的码，**并行逐位置**预测目标层所有码；对照组 = 完全相同架构、粗码槽换成可学习空向量（容量/优化严格匹配）；增益 = 对照组 CE − 条件组 CE。
**结果**（bits/码；bertB 与 bertHybrid 一致，此处 bertB）：

| 目标层 | q16 | q32 | q64 | q128 | q256 |
|---|---|---|---|---|---|
| 增益（无条件） | −0.46 | −0.31 | −0.16 | −0.05 | +0.08 |
| 增益（+前窗原文 prompt） | −0.09 | +0.08 | +0.02 | +0.03 | +0.06 |

补充：给 q256 的条件从 1 层粗码加到 5 层，CE 只降 0.03 bits；hybrid 的 q1 消融 = 0 增益（q1 无 planner 价值）；prompt 本身也只加 ~0.02 bits。

**分析**：**增益全在噪声内（≈0）**。结构性原因：残差量化的定义就是"第 k+1 层编码前 k 层的量化误差"，各层码天生近独立——这一个机制同时解释实验 4 的"无冗余"（正交）、实验 2 的平滑前缀曲线（累加）、和本实验的"不可预测"（正交的另一面）。早前泛用 AR 探针测到的 +2.05 bits"上下文增益"经此拆解，来源是**尺度内**相邻码相关，不是粗→细传递。已知限制：探针只有 4L×256d、以码 id 为输入（学不到码本几何、冷门码欠训练）——用累计反量化 latent 作输入的向量版探针是下一个待做项。

## 补充：q1 是"主题指纹"

`probe_planner.py` 的文档一致性检验：相邻窗口（多属同一篇文章）的 q1 码一致率远超随机——

| | bertPilot [1,2,4,256] | bertHybrid [1,8,...,256] |
|---|---|---|
| 相邻 / 随机一致率 | 41.5% / 0.4% | 58.7% / 2.6% |
| lift | **103.8×** | 22.6×（被密梯子稀释） |

q1 有语义身份价值，但（见实验 5）无预测价值。

---

## 总结论与下一步

Stage 0 的两半：**压缩成功**（近无损 + 层级可控 + 无冗余），**next-scale 可预测性失败**（结构性，两种条件设定下都 ≈0）。按 `docs/specs/need_next3.md` 的五条冻结标准：①边缘 ②✅ ③✅ ④✗ ⑤半 → **tokenizer 暂不冻结**。

Stage 0.5 候选（按优先级）：
1. 向量版探针：条件输入改为累计反量化 latent（对齐 VAR 真实接口 + 白送码本几何），确认"信息不存在"还是"id 读不出"；
2. tokenizer 训练加跨尺度耦合辅助损失（用累计 latent ≤k 预测第 k+1 层码），重训后复测实验 5；
3. planner 预测目标改为连续累计 latent（绕开 8192-way 近均匀码身份）。

## 踩过的坑（换环境请读）

- `datasets>=5` 的 `ds["text"]` 是懒加载列，逐行访问 O(表 chunk 数)，180 万行要数小时——用 `.data.column("text").to_pylist()`（已修）。
- VQ 死码复活不能用 EMA cluster_size 绝对阈值（K=8192 时会每轮重置 ~88% 码本），按窗口内原始使用次数判死（已修）。
- WikiText-103 训练集有 ~970 行伪标题正文（如 ` = Position ; GP = `），文档切分要求标题前后空行（已修）。
- 非前缀子集评估必须用 readout decoder，否则分布外失真（见实验 4）。

## 索引

全部 run 的提交命令与 Run ID：[`EXPERIMENTS.md`](EXPERIMENTS.md) ｜ 完整中文报告：[`docs/reports/`](docs/reports/) ｜ 原始结果 JSON：[`results/`](results/) ｜ 版本：`v0.1-pilot` → `v0.2-bert-line` → `v0.3-next3`
