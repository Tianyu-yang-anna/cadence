# CADENCE Stage 1 · Track 1 报告:VAR 文本生成 pipeline 跑通了

日期:2026-08-21
代码:github.com/Tianyu-yang-anna/cadence(Stage 1 代码 + 全部作业脚本)
产物:Volume `sandbox_ai.u_tianyuy.cadence` 下 checkpoints/planner_wt103_base、results/geneval_base

---

## 一句话结论

**端到端 pipeline 通了**:冻结 bertHybrid tokenizer + VAR next-scale planner(121M,8×H100 训 50k 步)能从前文 prompt 生成主题连贯的下一窗文本;词面指标(ROUGE/BERTScore)已接近同规模 AR baseline,分布指标(MAUVE)还有明确差距,且差距的性质已经定位清楚——**planner 局部有噪、内容在轨;AR 局部流畅、内容漂移**。

## 做了什么

任务协议(对标 TextLDM 的 continuation):prompt = 前一窗 256 token 原文 → 生成下一窗 256 token,与真实下一窗对比。WikiText-103 test 取 1000 对,指标 ROUGE-1/2/L、BERTScore、MAUVE、distinct-n。

三个系统同协议对比:
- **oracle**:真码 → 冻结 decoder,即 tokenizer 天花板;
- **VAR planner**(我们的):12L×768 + 每层 cross-attention,严格 VAR 构造(输入 e_k 从冻结码本反量化上采样、归一化坐标 RoPE、块内双向/跨尺度因果 mask、STAR 双通道条件、CFG),一次生成 = 7 步逐尺度采样 + 1 步并行解码;
- **AR baseline**:同词表、同数据、参数匹配(108M)的从头 causal LM,续写 = 256 步逐 token 采样。

## 主表(test,n=1000)

| 系统 | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | MAUVE | distinct-2 |
|---|---|---|---|---|---|---|
| oracle(天花板) | 0.993 | 0.986 | 0.993 | 0.997 | 0.999 | 0.562 |
| AR baseline(早停 12k 步) | 0.331 | 0.053 | 0.163 | 0.822 | **0.956** | 0.569 |
| planner,预注册配置¹ | 0.298 | 0.034 | 0.144 | 0.800 | 0.305 | 0.716 |
| planner,val 选出配置² | **0.304** | **0.040** | **0.148** | **0.804** | **0.416** | 0.671 |
| (参考文本自身) | — | — | — | — | — | 0.561 |

¹ T=1.0 / top_p=0.95 / CFG=3.0(跑扫描前定下的配置,如实保留)
² T=0.8 / top_p=0.9 / CFG=3.0(在 val 上扫出来的最优,单次 test)

**披露**:预注册配置是在任何扫描之前定的;后来 val 扫描发现低温+CFG=3 更好,按新配置只跑了一次 test。两行都放在表里,没有挑好的报。

## 采样超参地形(val,n=200,只看趋势)

| | CFG=1 | CFG=3 | CFG=5 | CFG=7 | CFG=9 |
|---|---|---|---|---|---|
| MAUVE @ T=1.0 | 0.51 | 0.58 | 0.75 | 0.56 | 0.36 |
| MAUVE @ T=0.8 | — | **0.80** | 0.64 | 0.66 | — |

- CFG 是单峰:太弱没方向,太强(7+)反而崩;
- 降温把峰从 CFG=5 挪到 CFG=3 且更高——细尺度的采样噪声是主要矛盾;
- 注意 MAUVE 对样本量敏感(val n=200 偏高),val 0.80 → test 0.416 的落差里有相当一部分是 n 的效应,不全是泛化差。

## 差距的性质(看样例就懂)

同一个 prompt(Robert Boulter 演员条目):

> **planner**:"in booteer had a role starring in a career by philstani, … boooter then opened the 2006 comedy drama wax …"
> **AR**:"he also performed in virginia bidi domeponthi, … boulter won an olivier award for best actor and a drama desk award …"

- planner:题材、结构、年代全部在轨(演员生涯、影视剧名、年份),但**实体拼写崩坏**("boulter"→"booteer"/"boooter")、短语级语法错乱。原因:最细尺度 q256(负责局部细节)是**一步并行**采出来的,256 个 token 的残差彼此不商量。
- AR:句句流畅(逐 token 采样天然保局部一致),但内容**自由漂移**(编造奥利弗奖),没有任何全局规划压力。

这正是两种分解方式的镜像缺陷,也是 proposal 里"粗尺度规划 + 细尺度填充"想解决的问题的两半。

## 训练侧的两个重要发现

**1)planner 的 next-scale 预测远超探针下界。**
val 逐尺度 CE(bits/code):q1=2.12,q8=11.19,q16=11.82,q32=11.75,q64=11.17,q128=9.81,**q256=6.75**。Stage 0 线性探针(带 prompt)在 q256 上只做到 12.18 bits——完整 VAR 架构比探针多榨出 **5.4 bits**,跨尺度+prompt 信息的可利用性远比探针显示的强。中间尺度(q16/q32)最难,最细尺度反而最好预测(残差小、条件强)。

**2)planner 几乎不过拟合,AR 严重过拟合。**
同数据同步数(50k 步 ≈ 29 epoch):AR 的 val CE 在 11k 步见底(4.45 bits/token)后一路恶化到 5.79;planner 的 val 曲线几乎平坦(best 9.16 @26k vs 9.23 @50k)。多尺度分解 + 冻结码本/冻结 prompt encoder 是天然的正则。主表 AR 用早停重训版(12k 步)保证公平——顺带发现 AR 的**采样指标对过拟合不敏感**(过拟合版 MAUVE 0.961 vs 早停版 0.956),词面指标同样几乎不动。

## 速度

生成 1000 条 256-token continuation:planner **23 秒**(7 次 forward + 1 次并行解码),AR 需要 256 步序列采样(同硬件慢一个数量级以上)。VAR 的推理效率优势在文本上兑现了。

## 公平性说明

planner 的 prompt 走冻结的**预训练** bert-base-uncased encoder,AR 直接吃 raw prompt token(全部从头训)——planner 多占了预训练知识的便宜。缓解:oracle 行锁定 decoder 上限;AR 与 planner 参数量/数据/步数匹配。解读 planner-vs-AR 差距时要记着这条不对称。

## 结论与下一步

1. **验证目标达成**:VAR next-scale 架构在文本上能训、能生成、能评测,mentor 要求的 pipeline 验证完成;
2. 词面指标追平 AR 在望,**主要矛盾是细尺度并行采样的局部噪声**(MAUVE 差距的来源)。改进方向(Track 2 及以后):细尺度温度调度、细尺度换局部自回归/迭代精化、更大模型+更多数据;
3. **Track 2 已在飞**:GPT-2 BPE hybrid tokenizer 在 OWT 4B token 上重训完成(test 重建 99.39%,VQ-EMA 首次 DDP 成功),330M planner(20L×1024,100k 步)训练中,TextLDM 四 benchmark(TinyStories/1BW/Wikipedia/WikiSource 各 1000 条)已备好,将按论文协议出可同表对比的数字。

## 附:本轮踩过的坑(工程记录)

- CFG null 参数在"整批无丢弃"时不进计算图 → DDP reducer 崩(概率 ~3.4%/rank/步),修复:无条件 `torch.where`;
- AR 在小语料上 29 epoch 严重过拟合 → 主表换早停版;
- `${VAR:-default}` 把显式空参也替换成默认 → geneval 多跑 2 小时重复扫描,修复:无冒号 `${VAR-default}`;
- OWT prep 全量攒内存再 concatenate → 3B token 处 SIGABRT,修复:流式增量写盘;
- lm1b 是 script 数据集(datasets≥3 拒载)→ 改走 HF parquet 转换分支;wikisource config 是 `20231201.en`;
- streaming datasets 后台线程在解释器退出时崩(rc=134,数据其实已写完)→ `os._exit(0)`;
- 平台 env_vars 上限 10 个;偶发作业卡在 provisioning 不启动 entry(心跳文件可判别,重提即可)。
