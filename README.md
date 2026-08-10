# CADENCE Stage 0 — 文本多尺度 VQ-VAE

把 256 个词的文本压成一串**从粗到细的离散码**（比如 1 个"主题码" + 8 个段落级码 + … + 256 个词级码），再用一个双向 decoder 一次并行还原全文。这是 CADENCE 项目（把图像领域的 VAR "先轮廓后细节"生成范式搬到文本）的第一阶段：先造出这套"多尺度文本压缩码"，并验证它撑不撑得起后面的生成模型。

```
256 个 token ──encoder──▶ 256 个向量 ──多尺度残差量化──▶ 粗码+细码 ──decoder──▶ 还原 256 个 token
                                        (例: 8+16+32+64+128+256 个离散码)
```

## 三轮实验后的结论（人话）

1. **压缩这一半彻底成功**：还原率 99.4–99.9%（几乎无损），且每一层码都在干活（藏掉任何一层，还原率都会掉）。
2. **层级不会自己长出来，但能造出来**：必须在训练时随机"没收细码"逼模型（scale dropout 0.5）；schedule 用密梯子 [8,16,...,256] 最好——120 个粗码能还原 52% 的词，248 个能还原 84%。
3. **坏消息（最重要的发现）**：**知道粗码，猜不到细码**——我们的量化是"残差式"的，每层专门编码上一层没压进去的部分，层与层天生互相独立。这对压缩是优点（不浪费），对"先生成粗码、再生成细码"的 Stage 1 规划器是致命的：严格测量下，粗码对预测细码的帮助 ≈ 0，连把上文 prompt 给它也一样。
4. **所以 tokenizer 暂不冻结**，需要先做 Stage 0.5：给训练加"跨尺度耦合"损失，或改用连续 latent 作为规划器的预测目标。

完整故事：`docs/reports/` 两份中文报告（第一二轮 / 第三轮）；逐实验索引：[`EXPERIMENTS.md`](EXPERIMENTS.md)；英对照详细日志：`docs/RESULTS_LOG.md`；冻结判定：`results/hybrid_schedule_summary.md`。

## 仓库结构

```
configs/          三个 yaml（GPT-2 版 / BERT 版 / CPU 调试版）；消融全靠 --set 改，不用改代码
models/           模型本体：双向 transformer、VQ-EMA、多尺度残差量化、组装
data/             WikiText-103 下载+分词成 .bin（GPT-2 或 BERT 词表）、数据加载
train_vqvae.py    训练（bf16、量化旁路热身、scale dropout、断点续训）
eval_vqvae.py     评估（全量重建 + 截断表 + 码本健康 + 样例还原）
probe_planner.py            探针 1：码的可预测性 + q1 主题指纹
probe_next_scale.py         探针 2：严格版"粗码能否预测细码"（含对照组）
probe_next_scale_prompted.py 探针 3：加上 prompt 再测一次
finetune_subset_readout.py  给"藏码评估"训一个会读任意子集的 decoder（修分布外失真）
eval_scale_subsets.py       藏码评估：留一尺度 / 单尺度 / 邻居组合
analyze_runs.py   跨 run 对比表
jobs/             Databricks 作业脚本（提交、引导、各实验入口）
tests/            56 个单元测试（含数学不变量、无信息泄漏检查）
results/          全部实验的结果 JSON + 汇总
docs/             实验需求文档（specs/）、中文报告（reports/）、详细结果日志
```

## 快速开始

**本地跑通（Mac，CPU，几分钟）**：
```bash
uv venv && uv pip install --python .venv/bin/python -r requirements-local.txt
.venv/bin/python -m pytest tests/ -q                # 全部测试
.venv/bin/python train_vqvae.py --config configs/vqvae_tiny_cpu.yaml --resume none   # 微型端到端
```

**准备数据（本地 ~3 分钟，别在远端节点跑，见下面的坑）**：
```bash
python data/prepare_wikitext.py --tokenizer bert-base-uncased --out /tmp/wikitext103_bert_bins
```

**训练/评估/探针**：单卡即可（~70M 参数）。我们在 Databricks serverless GPU 上跑，提交命令见 `EXPERIMENTS.md`；如果换环境，直接运行 `train_vqvae.py`/`eval_vqvae.py` 即可，路径都在 config 里。失败或超时重跑同一条命令 = 自动断点续训。

## 换 schedule / 做消融

全部是命令行参数，不改代码：
```bash
python train_vqvae.py --config configs/vqvae_wikitext_bert.yaml \
    --set "quantizer.scales=[8,16,32,64,128,256]" \
    --set train.scale_dropout_p=0.5
```

## 踩过的坑（换环境前请读）

- `datasets>=5` 的 `ds["text"]` 是懒加载列，180 万行的表逐行访问要**几个小时**——一定用 `.data.column("text").to_pylist()` 一次取出（`data/prepare_wikitext.py` 已修）。
- VQ 死码复活不能拿 EMA cluster_size 卡绝对阈值（大码本下会每轮重置 ~88% 的码），要按"窗口内原始使用次数"判死（`models/vq_ema.py` 已修）。
- 评估"藏掉某层码"时 decoder 处于分布外，会自信地胡说——先用 `finetune_subset_readout.py` 训个读出头再测。
- WikiText-103 训练集有 ~970 行长得像标题的正文（如 ` = Position ; GP = `），切文档必须要求标题前后是空行。

## 版本

`v0.1-pilot`（可行性）→ `v0.2-bert-line`（层级形状）→ `v0.3-next3`（冻结前验证，结论：暂不冻结）。各版本做了什么见 `EXPERIMENTS.md` 末尾。
