# `results/` —— 全部实验产物

每个实验臂一个目录，里面是它在每个 benchmark 上的生成文本与指标。**报告里的每一个
数字都能追到这里的一个文件。**

## 布局

```
results/benchgen_<run_name>/
    gens_<bench><TAG>.jsonl           # 生成文本（每行一条）
    gens_<bench><TAG>.metrics.json    # 该文件的指标
```

- `<bench>` ∈ `wikipedia` / `wikisource` / `tinystories` / `lm1b`，前缀 `sel_` 表示
  **选择集**（n=250，与 test 不相交，用于选采样配置）；无前缀是 **test**（n=1000，
  每个配置只打一枪）。
- `<TAG>` 标识具体配置，例如 `_finalSEG`（段轴主行）、`_finalPOSF`（位置轴对照）、
  `_sw8k1`（chunk 数扫描 C=8,K=1）。臂 ↔ TAG ↔ 含义的完整登记表在
  `tools/build_master_table.py` 的 `ROWS`，那里每行还带着限定它的 caveat。

## `gens_*.jsonl` 的字段

| 字段 | 含义 |
|---|---|
| `index` | benchmark 文件里的绝对行号。**采样种子由它派生**，所以同一 index 在不同臂之间是配对的 |
| `prompt` | 输入前缀。同一 benchmark 的同一 index，在所有臂之间逐字节相同 |
| `reference` | 该 prompt 的真实后续。同上，跨臂相同 |
| `generated` | 该臂生成的后续。**这是唯一随臂变化的字段** |

`prompt`/`reference` 在几十个臂之间重复存储，是为了让每个文件自包含、可以单独喂给
`tools/mauve_variance.py` 之类的工具重打分。git 的 delta 压缩会吃掉这部分冗余。

**两处已知的、无害的不一致**（不要当成 bug 去"修"）：

- **`benchgen_{bd3lm,mdlm,ssdlm}_owt2` 的 34 个文件没有 `index` 字段。** 这三家 baseline
  走的是移植过来的各自 harness，写出时不带绝对行号。它们的行**顺序**仍与 benchmark
  文件一致，所以按行号配对依然成立；只是不能像我们自己的生成那样从字段直接读出 index。
- **`benchgen_planner_owt/gens_*_dd.jsonl` 4 个文件没有配套的 metrics.json。** 那是
  Stage 1 的 decoder-denoise 臂，两次被墙钟压成欠训后弃用、从未评测。生成文本保留以
  记录，但**没有任何报告数字依赖它们**。

完整性检查（391 个有 metrics 的文件，行数 vs `metrics.json` 的 `n`）：**全部一致，
零不匹配**。

## 怎么用

重建全部对照表（不要手抄数字）：

```bash
python tools/build_master_table.py     # -> docs/reports/MASTER_TABLES.md + results/master_table.csv
```

重新打分一批已有生成（不重新生成，用于换 MAUVE 簇数等估计量稳健性检查）：

```bash
python tools/mauve_variance.py --gen results/benchgen_.../gens_wikipedia_finalSEG.jsonl ...
```

## 读这些数字之前必须知道的三件事

1. **`sel` 与 `test` 的 MAUVE 不可放进同一句话。** `eval_generation.py` 把 mauve 的
   `num_buckets` 留给 `'auto'` = n/10，所以 sel 用 ~25 簇、test 用 ~100 簇。实测把
   同一份 1000 行 test 改用 25 簇重算会**翻转排序**（粗化幅度依臂而异，不保序）。
2. **n=250 上 MAUVE 的 bootstrap 标准差实测是 3.2~5.3 分。** 单看一格 sel 的高低没有
   意义；扫描取最大值本身就抬高约 1.42σ。
3. **有些行已作废或带限定**，例如 `b2sp` 的全部 `lr` 解码行（训练/推理不匹配）、
   `_final4`（非严格 2B 预算）。文件仍保留以记录推理链，**判读前先看
   `tools/build_master_table.py` 里对应行的 note，以及 `docs/reports/` 的撤回说明**。

行分片改造（`generate_prefix.py --shard/--nshards`，按绝对行号派生每行种子）之前注册的
test 行，任何新跑都不可逐比特复现；已注册的指标作为测量仍然有效。
