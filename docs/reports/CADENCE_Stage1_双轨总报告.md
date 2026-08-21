# CADENCE Stage 1 双轨总报告:VAR 文本生成从跑通到对标

日期:2026-08-22
代码:github.com/Tianyu-yang-anna/cadence(tag `v0.5-stage1-track1` 起为 Stage 1)
单轨详细报告:`CADENCE_Stage1_Track1_报告.md`(同目录)· 英文数据:repo 内 `results/stage1_track{1,2}_summary.md`

---

## 一页结论

Stage 1 的任务是:搭出 VAR next-scale 架构、训一个文本生成模型、验证 pipeline(mentor 指示),并对标 TextLDM(arXiv 2605.07748)。两轨都完成了:

1. **Track 1(WikiText-103,121M)**:pipeline 端到端跑通。词面指标接近同规模 AR(差 ~0.03),MAUVE 有差距,且差距性质定位清楚——**规划在工作,渲染是短板**(最细尺度 256 个码一步并行采样导致局部噪声)。
2. **Track 2(OpenWebText 4B token,336M)**:按 TextLDM 论文协议在四个 benchmark 上评测,**总体坐进 GPT-2-137M / TextLDM-114M 一档,而训练量只有对方的零头**;Wikipedia 的 MAUVE(15.1)是整张表最高;短句 benchmark(1BW)最差、原因已诊断(prompt 长度出分布)。
3. **"想法有没有问题"的规模响应证据是正面的**:数据 ×30、模型 ×3 后,最细尺度预测从 6.75 → **4.03 bits/code**,val 曲线到 100k 步仍在下降,benchmark 质量进入预训练 GPT-2 档。目前没有任何证据否定层级假设;瓶颈(细尺度渲染、短 prompt OOD)都有明确、便宜的修法。

## Track 1:pipeline 验证(细节见单轨报告)

test 1000 条窗口续写,三系统同协议:

| 系统 | R-1 | R-2 | BERTScore | MAUVE |
|---|---|---|---|---|
| oracle(tokenizer 天花板) | 0.993 | 0.986 | 0.997 | 0.999 |
| AR baseline(早停,108M) | 0.331 | 0.053 | 0.822 | 0.956 |
| VAR planner(121M,val 选优采样) | 0.304 | 0.040 | 0.804 | 0.416 |

要点:planner 主题在轨但局部噪(细尺度并行采样),AR 流畅但内容漂移——镜像缺陷;planner q256 预测比 Stage 0 探针下界低 5.4 bits(架构远超线性探针);planner 几乎不过拟合而 AR 29 epoch 严重过拟合;1000 条生成只要 23 秒(7 步并行 vs AR 256 步)。

## Track 2:TextLDM 对标

**Tokenizer 重训**:GPT-2 BPE(cased、无损)+ hybrid schedule,OWT 4B token,100k 步——**test 重建 99.39%**,达到论文 Table 3 区间(97.5–100%);VQ-EMA 首次 8 卡 DDP(EMA 统计 all-reduce)一次成功,码本无坍缩。

**Planner**:335.7M(20L×1024),100k 步 ≈ 1.6 epoch,~6.5 小时,零故障零过拟合。逐尺度 val CE:q1=3.06 / q8=11.03 / q16=11.65 / q32=11.53 / q64=10.67 / q128=8.48 / **q256=4.03**。

**Benchmark 主表**(论文单位 ×100;论文行为原文引用;完整表和 caveats 见 repo):

| 模型 | WikiSource R-1/MAU | Wikipedia R-1/MAU | TinyStories R-1/MAU | 1BW R-1/MAU |
|---|---|---|---|---|
| GPT-2 137M(预训练) | 31.1 / 35.3 | 23.3 / 7.9 | 31.8 / 1.04 | 13.4 / 0.45 |
| TextLDM 114M | 33.0 / 21.6 | 27.5 / 8.9 | 36.7 / 1.00 | 10.3 / 0.77 |
| TextLDM 328M | 33.1 / 27.6 | 27.6 / 10.5 | 37.1 / 1.13 | 10.8 / 0.79 |
| TextLDM 768M | 37.5 / 32.7 | 38.9 / 10.1 | 39.7 / 1.51 | 21.4 / 0.80 |
| **CADENCE 336M** | 29.3 / 8.4 | 24.3 / **15.1** | 30.4 / 0.58 | 9.3 / 0.64 |

诚实解读:

- **强项**:Wikipedia MAUVE 全表第一(15.1 vs 对方最好 10.5),R-1 高于预训练 GPT-2-137M;1BW 的 MAUVE 也超 GPT-2-137M。长而多样的 OOD 文本上,分布质量是真实信号。
- **弱项**:R-2 系统性偏低(细尺度采样噪声直接毁 bigram——和 Track 1 同一个病根);WikiSource 的 MAUVE 差距明显;1BW 全面最弱。
- **1BW 差的原因确诊**:它是句子级语料,prompt 只有十几个词,而 planner 训练时只见过 256-token 定长 prompt——严重出分布,样例接近词沙拉。修法便宜:混合长度 prompt 训练。TextLDM 自己也在这个 benchmark 上掉链子(原因类似:长度失配)。
- **对比 caveats**(必须说):样本集是按协议重抽的(非论文原 1K 条);tokenizer 不同(GPT-2 vs Qwen3);我们 256-token 窗链式 vs 对方 1024 原生;训练量差距巨大(我们 100k 步 vs TextLDM DiT 2M 步,GPT-2 是 WebText 全量预训练)。跨表数字当"量级参照",不当精确对比。

## 对"想法有没有问题"的回答(承接上次讨论)

三个判据,两个已有答案:

1. **规模响应(✅ 正面)**:q256 CE 6.75→4.03,val 到 100k 步仍在降,benchmark 进 GPT-2 档——差距随规模收缩,支持"工程短板"而非"根本缺陷"。
2. **渲染/规划隔离(未做,已排队)**:给真实粗尺度码只采最细尺度,一个 1×H100 作业即可归因剩余差距。
3. **长文本全局一致性(未做)**:链式生成 vs AR 的主题漂移对比——层级假设真正的主场,也是对 TextLDM 的差异化卖点(定长 DiT 做不到任意长度)。

中间尺度熵高(~11.5/13 bits)的隐忧仍在(粗计划对中层实现约束弱),但 Track 2 没有让它恶化,且它与"benchmark 中游+Wikipedia MAUVE 第一"共存——更像语言本身的多样性而非层级失效。

## 下一步(按性价比排序)

1. **混合长度 prompt 训练**(修 1BW 短板,数据管线现成,重训 planner 即可)
2. **oracle-coarse 渲染诊断**(一个作业,定量归因)
3. **细尺度渲染修复**:逐尺度温度调度(零训练成本,先扫)→ q256 局部自回归/迭代精化(改架构)
4. **长文本一致性评测**(卖点实验,建议进下一次组会讨论)

## 工程账(本轮新增)

全部修复已进 repo(README Gotchas 有完整清单):CFG null 参数 DDP 必进图;`${VAR:-}` vs `${VAR-}` 空参语义;OWT prep 流式写盘(内存 SIGABRT);lm1b 走 parquet 分支 + wikisource config 版本;streaming 线程 teardown 崩溃 `os._exit(0)`;平台 env_vars≤10、偶发 provisioning 挂起(心跳文件判别)。产物:tokenizer/planner/AR 各 ckpt 在 Volume `checkpoints/`,指标在 `results/geneval_base` 与 `results/benchgen_planner_owt`,repo 内有全部 JSON 副本。
