# CADENCE 交接文档(给新会话/新接手人)

更新:2026-08-24。本文是唯一需要通读的入口;读完即可无缝接手。

## 0. 一句话:项目是什么、走到哪了

把 VAR(视觉自回归 next-scale prediction, NeurIPS 2024)适配到文本生成:冻结的多尺度残差 VQ-VAE 把 token 窗口压成粗→细码层级,planner 逐尺度生成码、decoder 一次并行解码。对标论文 = **TextLDM(arXiv 2605.07748)**,协议 = 四 benchmark 续写(ROUGE/BERTScore/MAUVE)。

**当前旗舰成绩**(336M planner、1024 原生窗、s5 采样调度,test 4×1000,单位 ×100):
**Wikipedia MAUVE 26.0 = TextLDM 全表最好成绩(10.5)的 2.5 倍;R-1 26.2 超全部 GPT-2 与 TextLDM-114/328M**。短板:WikiSource MAUVE 15.2(GPT-2 系 35+)、全线 R-2 偏低。完整终表:`results/stage2_scaleup_summary.md` §6。

**当前状态:全部评测收官,无活动作业;768M 冻结(用户要改设计后再跑),其全部物料已就绪。**

## 1. 资产地图

| 资产 | 位置 | 说明 |
|---|---|---|
| 代码(唯一真源) | github.com/**Tianyu-yang-anna/cadence**,main 分支 | 私有;推送凭证在 macOS 钥匙串:`security find-internet-password -s github.com -w`;无 gh CLI |
| 本地 clone | `/Users/tianyuy/cadence` | 与远端同步;127 个单测:`.venv/bin/python -m pytest tests/ -m "not slow" -q` |
| 版本标签 | v0.1-pilot…v0.4-probe-correction(Stage 0)、v0.5/v0.6(Stage 1) | Stage 2 在 main 尾部 |
| Proposal 与规范 | `~/Documents/cadence/need.md`(Stage 0 权威规范)+ proposal PDF(同目录) | |
| 中文报告 | `~/Documents/cadence/CADENCE_Stage{0,1,2}*报告.md`(repo `docs/reports/` 有副本) | Stage2 报告 = 最新总结 |
| 英文数据档案 | repo `results/stage2_scaleup_summary.md`(§1-6)+ `docs/RESULTS_LOG.md`(rounds 1-11) | 全部原始指标 JSON 在 `results/` 子目录 |
| Databricks Volume | `/Volumes/sandbox_ai/u_tianyuy/cadence/{data,checkpoints,results,logs,status,envs}` | profile `tianyuy-ws`;查看:`databricks fs ls dbfs:/Volumes/sandbox_ai/u_tianyuy/cadence/status -p tianyuy-ws` |
| 会话记忆 | `~/.claude/projects/-Users-tianyuy/memory/cadence-stage0.md` | 与本文互为备份 |

## 2. Volume 上的关键产物

**数据(data/)**:`wikitext103{,_bert}`(Stage 0/1)、`owt_gpt2`(4B)、`owt9_gpt2`(9.5B)、**`c4_gpt2`(40.0B token/83.6M docs,768M 备料,已合并)**;码:`codes_hybrid`(WT103)、`codes_owt`(4B)、`codes_owt9`(9.5B/256 窗)、`codes_owt9_1024`(9.5B/1024 窗);benchmark:`benchmarks/{tinystories,lm1b,wikipedia,wikisource}.jsonl`(测试 4×1000)+ `sel_*.jsonl`(不相交选择集 4×250)。

**Checkpoints(checkpoints/)**:
| 名字 | 内容 |
|---|---|
| `vqvae_wt103_hybrid` | Stage 0 冻结 tokenizer(BERT,99.44%) |
| `vqvae_owt_gpt2hybrid` (+`_dd`) | 256 窗 GPT-2 tokenizer 99.39%(+去噪 decoder) |
| `vqvae_owt9_1024_d5` | **1024 窗 tokenizer,99.76%,当前主力** |
| `planner_wt103_base` / `ar_wt103_es12k` / `ar_plan_wt103_base` | Stage 1 Track1 planner/早停 AR/计划条件 AR |
| `planner_owt` / `planner_owt_v2` | 256 窗 planner(原始/数据修复版) |
| `planner_owt1024` (+`_mg`) | **旗舰 1024 planner**(+MaskGIT 微调版,test 持平) |

**结果(results/)**:`geneval_base`(Track1)、`benchgen_planner_owt{,_v2,1024,1024_mg}`(全部 benchmark 指标+生成文本)、`paircheck`、`scale_info`、`schedsweep*`——repo `results/` 有全部指标 JSON 副本。

## 3. 作业系统速查

```bash
cd ~/cadence   # 仓库必须干净(sgcli 用 git 快照);env_vars ≤10 个(平台自占 1)
bash jobs/submit.sh <stage> <suffix> <timeout_min> <gpu> KEY=VAL...
# stage: train/planner/arbase/arplan/dumpcodes/dumpshard/geneval/benchgen/
#        schedsweep/owtprep/c4prep/c4merge/codemerge/decdd/maskgit/paircheck/scaleinfo
# gpu:   1xh100(配额4) | 8xh100 | 16xh100 | 32xh100 | 64xh100(多节点)
# 失败/超时(墙=600min):原样重提 = 断点续训(续训抽全新数据排列)
# 监控:databricks jobs get-run <id> -p tianyuy-ws;进度/日志/标记在 Volume status|logs/
```
推理默认:`SCHED_PRESET=s5`(11 尺度)或 `hc7`(7 尺度);`TOK_FULL`/`PLANNER_FULL` 决定加载哪套模型(benchgen 里已透传 tokenizer_run_dir)。多节点四坑已解(见 `jobs/planner_entry.sh` 注释):平台 MASTER_PORT/3600s join 超时/--local-addr 数字 IP/静态会合。

## 4. 三阶段科学结论(足以支撑写作的最小集)

1. **Stage 0**:近无损重建(99.4%+);层级需 scale_dropout=0.5;密梯子最优;**探针接口教训**——planner 输入必须严格 VAR e_k(冻结码本反量化上采样),偏离即测出假阴性(结论曾被反转);
2. **Stage 1**:pipeline 通;差距性质 = 渲染非规划(planner 主题在轨局部噪 vs AR 流畅但漂移);planner 几乎不过拟合而 AR 严重过拟合;7 步并行生成 1000 条 23 秒;
3. **诊断波**:跨文档污染 40.1%(修复=靶向收益,1BW +16%);中尺度有信息(量化器无罪);**粗计划对 AR 渲染器值 1.36 bits/token(层级假设因果证据,plan-then-write 绿灯)**;
4. **Stage 2**:**逐尺度采样调度 = 最大零训练杠杆**(粗热细近 argmax;两模型两梯子复现);**1024 窗证伪"层级浅"**(粗尺度条件熵 3.4-6.5 bits vs 256 窗近满)→ Wikipedia MAUVE 21.9→26.0;MaskGIT 精化在好调度之上无增益、best-of-4 换多样性买词面(两个诚实中性结果);**协议铁律:采样配置必须在同分布不相交选择集上选**(两次迁移失败教训;`prepare_benchmarks --skip`,seed 只动切点不动文档)。

## 5. 冻结中的 768M(物料 100% 就绪,等设计改动)

就绪:40B C4 语料(`data/c4_gpt2`)、配置 `configs/planner_owt1024_768m.yaml`(28L×1536≈800M、gpt2-medium encoder、250k 步、micro 2)、64 卡通道、分片码导出管线。**点火步骤**(设计定稿后):
```bash
# 1) 码导出(8 分片并行,~3h;若 tokenizer 有改动则先重训 tokenizer)
for s in 0..7: jobs/submit.sh dumpshard ds$s 300 8xh100 SHARD=$s NSHARDS=8 \
  RUN_NAME=owt9tok FULL_NAME=vqvae_owt9_1024_d5 CONFIG=configs/tokenizer_owt9_1024.yaml \
  DATA_NAME=c4_gpt2 CODES_NAME=codes_c4_1024
# 2) jobs/submit.sh codemerge ... CODES_NAME=codes_c4_1024 NSHARDS=8
# 3) jobs/submit.sh planner c4768m 600 64xh100 RUN_NAME=c4768m FULL_NAME=planner_c4_768m \
#      TOK_FULL_NAME=vqvae_owt9_1024_d5 CONFIG=configs/planner_owt1024_768m.yaml \
#      DATA_NAME=c4_gpt2 CODES_NAME=codes_c4_1024   # ~35h,超时原样重提续训
# 4) 选择集调度扫描(GRID_SET=11)→ 胜者 4×1000 终评(benchgen)
```
数据红线:250k 步 ≤4 epoch 需 ≥40B token(已满足)。若改码本/梯子 → tokenizer 必须重训(pilot 25k 步先行),码导出随之重做;若只改 planner 形状/超参 → 直接从第 1 步走。

## 6. 悬置决策与待办

1. **768M 设计改动**(用户持有)——改什么等用户定;
2. **plan-then-write 路线**:+1.36 bits 证据在库;下一实验 = AR 渲染器吃 planner 生成计划的端到端原型(~1 天),建议组会拍板;
3. WikiSource MAUVE 差距、R-2 偏低:剩余手段 = 规模(768M)/REPA 式对齐(研究性)/plan-then-write;
4. 未做:长文链式一致性评测(vs TextLDM 定长的卖点实验);decdd-1024 四次节点故障弃用(非阻塞待查);
5. 平台清理项(闲时):Volume 上 c4_gpt2/shard* 与 codes 分片目录可在合并验证后删除省空间。

## 7. 新会话接手清单

1. 读本文 + `results/stage2_scaleup_summary.md`(数据)+ 记忆文件(坑清单全集);
2. 验证环境:`cd ~/cadence && git pull && .venv/bin/python -m pytest tests/ -m "not slow" -q`(应 127 通过);
3. 验证平台:`databricks fs ls dbfs:/Volumes/sandbox_ai/u_tianyuy/cadence/status -p tianyuy-ws`;
4. 所有历史作业 run-id 与逐轮实验记录:`docs/RESULTS_LOG.md`;
5. 用户偏好:中文回复、"写人话"、效果优先、报告存 `~/Documents/cadence/`、重大路线变更先给证据由用户/mentor 拍板。
