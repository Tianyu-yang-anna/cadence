# MaskGIT 超参数：设在哪里、当前值是什么、NFE 怎么算

（补这份文档的直接原因：`.job-*.yaml` 提交记录被 gitignore，所以仓库里以前**看不到**
每个臂到底用了什么超参。本文 + `tools/build_master_table.py` 的 `ROWS` 登记表现在是
权威来源。）

## 1. 解码侧超参数在哪里设

链路：`jobs/submit.sh benchgen ... SAMPLE_MODE=<spec>` → `jobs/benchgen_entry.sh`
（第 69/140 行，原样透传）→ `generate_prefix.py --sample_mode <spec>`（解析在
~第 142-160 行）→ `models/prefix_planner.py::generate()` 路由到各解码函数。

`<spec>` 的完整语法（`<scales>` 是逗号分隔的尺度下标，`all` = 0..10）：

| spec | 含义 | 超参数 |
|---|---|---|
| `seg:<scales>:<K>` | 段轴 MaskGIT（`_sampler_seg_scale`） | K = 每尺度承诺轮数，K=S=4 为完全顺序化 |
| `pos:<scales>:<K>` | 位置轴 MaskGIT（`_sampler_pos_scale`） | K = 每尺度承诺轮数（cosine 日程） |
| `ar:<scales>` | 严格逐位置左到右（KV-cache） | 无 |
| `lr:<scales>:<C>:<K>` | 约束左到右（chunk 顺序 × chunk 内位置 MaskGIT） | C = chunk 数，K = chunk 内轮数 |
| `lrseg:<scales>:<C>:<Kseg>` | **2D MaskGIT**（chunk 顺序 × chunk 内段轴置信度） | C = chunk 数，K_seg = chunk 内段轮数（≤S） |

每尺度的温度/top-p/CFG 由 `SCHED_PRESET`（`benchgen_entry.sh` 第 52-67 行）给出；
主线用 `p5hot7`（CFG 梯子 7,…,7,4,2,1.5）。

## 2. 两个"原始配置"的准确定义（mentor 引用的 48 和 88 就是这两个）

| 名字 | spec | 采样器前向 | backbone 前向 |
|---|---|---|---|
| **原始段轴（主行 `_finalSEG`）** | `seg:all:4` | 2×4×11 = **88** | 22 |
| **原始位置轴（`_finalPOSF`）** | `pos:8,9,10:8` | 2×8×3 = **48** | 22 |

NFE 公式（系数 2 = CFG 双分支；backbone 恒为 2×11=22，采样器只在冻结 hidden 上迭代）：

```
seg   : 每尺度 2·K              （K ≤ S=4）
pos   : 每尺度 2·K
lr    : 每尺度 2·min(C,l)·K
lrseg : 每尺度 2·min(C,l)·K_seg
```

这些公式有门禁测试钉死（`tests/test_sampling_transformer.py` 的
`test_lr_nfe_*` / `test_lrseg_nfe_*`：instrument 计数，逐格断言）。

## 3. 训练侧超参数在哪里设

链路：`jobs/submit.sh planner ... EXTRA_ARGS=...` → `jobs/planner_entry.sh` →
`finetune_prefix_maskgit.py`。关键旗标：

| 旗标 | 作用 |
|---|---|
| `--mask_mode` | 训练揭示模式，**必须与解码配对**：`sampler_seg`↔`seg`、`sampler_pos`↔`pos`、`sampler_causal`↔`ar`、`sampler_lr`↔`lr`、`sampler_lrseg`↔`lrseg` |
| `--mask_scales` | 采样器训练的尺度集（解码只能在其子集上跑，有断言） |
| `--chunks` | `sampler_lr`/`sampler_lrseg` 的 chunk 网格；lrseg 每样本从其约数 {1,2,4,8,…} 里抽 C，使扫描里的每个推理 C 都在分布内 |
| `--lr_supervise` | `sampler_lr` 的损失范围（chunk=只监督当前 chunk，默认） |
| `--set planner.depth_ar=false` | 冻结 depth_projs（零初始化 → 精确并行段头）。注意 `sampler_seg`/`sampler_lrseg` **无条件**冻结 |

**训练/解码不配对 = 本项目已犯过两次的 E1 类错误**（`b2sp` 用 `lr` 解码、
`sampler_mix` 污染），任何新解码模式必须先加配套 mask_mode + 门禁测试再上。

## 4. 2026-09-04 起的 2D 扫描（当前波）

训练臂 `b2s2d`：`b2sq1` + 1000 步 `sampler_lrseg`，`--chunks 8`、
`--mask_scales 8,9,10`、depth 冻结（与 `b2sg`/`b2spf`/`b2slr3` 同父同预算，单变量）。

解码扫描（全部 `lrseg:8,9,10:<C>:<Kseg>`，两档 iso-NFE）：

| 档位 | 格 | 每尺度采样器前向 | 3 尺度合计 | 对应 mentor 的哪一组 |
|---|---|---|---|---|
| **P=8 档（合计 48）** | C=2,K=4 | 16 | 48 | (1) 段轴原样，位置轴粗化 |
| | C=8,K=1 | 16 | 48 | (2) 位置轴原粒度（8 档），段轴粗化（整 chunk 一次并行） |
| | C=4,K=2 | 16 | 48 | (3) 两轴都粗化 |
| **P=4 档（合计 24，< 48 与 88）** | C=2,K=2 | 8 | 24 | 两轴都粗化（更低 NFE） |
| | C=4,K=1 | 8 | 24 | 位置轴为主 |
| | C=1,K=4（锚点） | 8 | 24 | 纯段轴@细尺度（= `seg:8,9,10:4`，同臂 C=1 退化，逐比特等价有门禁测试） |

另加 `seg:8,9,10:4` @ `b2sg`（跨臂锚点，兼答第二轮审计的 footprint 问题）。
所有格 backbone 恒 22。锚点 C=1,K=4 与 P=4 档同价，是"2D 是否优于纯段轴"的直接对照。
