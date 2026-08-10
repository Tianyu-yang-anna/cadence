# 实验总索引（人话版）

每个实验：干了什么、怎么跑的、结果在哪、一句话结论。按时间分三轮。
所有训练默认：WikiText-103、256-token 窗口、~70M 参数模型、50k 步、1×H100 约 6.5 小时。
结果 JSON 都在 `results/`，两份完整中文报告在 `docs/reports/`。

---

## 第一轮：验证基本可行性（GPT-2 词表）

| run | 设置 | 一句话结论 | 结果文件 |
|---|---|---|---|
| `base` | scales [1,2,4,256]，**不加** scale dropout | 重建 99.79%，但粗码是摆设（层级不会自己长出来） | `results/eval_test_base.json` |
| `sd05` | 同上 + dropout 0.5 | 粗码开始带信息（7 码解出 6.7%），层级能被逼出来 | `results/eval_test_sd05.json` |

复现命令（在仓库根目录，Databricks 环境见 README）：
```bash
bash jobs/submit.sh train base 480 1xh100 RUN_NAME=base CONFIG=configs/vqvae_wikitext.yaml
bash jobs/submit.sh train sd05 480 1xh100 RUN_NAME=sd05 CONFIG=configs/vqvae_wikitext.yaml \
    EXTRA_ARGS="--set train.scale_dropout_p=0.5"
```

## 第二轮：找层级的正确形状（换 bert-base-uncased 词表）

四个 schedule 消融 + 三个单因子 + 一个基准，全部 dropout 0.5：

| run | scales | 一句话结论 | 结果文件 |
|---|---|---|---|
| `bertPilot` | [1,2,4,256] | 第二轮的参照系；重建最高 99.85% | `results/eval_test_bertPilot.json` |
| `bertD` | [1,4,16,64,256]（稀梯子） | 85 码解 31%——比 pilot 好但不如密梯子 | `results/eval_test_bertD.json` |
| `bertA` | [4,8,...,128,256] | 124 码解 40.6% | `results/eval_test_bertA.json` |
| `bertB` ⭐ | [8,16,...,128,256] | **每个预算点都最优**：120 码解 52.5%、248 码解 84.4% | `results/eval_test_bertB.json` |
| `bertC` | [1,2,4,...,128,256]（全梯子） | 127 码解 38.7%——超粗尺度 1/2/4 是低效台阶 | `results/eval_test_bertC.json` |
| `bertPhi` | pilot + φ 卷积 | 没用，弃用 | `results/eval_test_bertPhi.json` |
| `bertSepCB` | pilot + 每尺度独立码本 | 只赚 0.3 个点，不值 4 倍参数 | `results/eval_test_bertSepCB.json` |
| `bertP75` | pilot + dropout 0.75 | 和 0.5 一样——瓶颈是容量不是压力 | `results/eval_test_bertP75.json` |

```bash
# schedule 消融示例（bertB）；其它组只换 scales 和 revival.interval（=25×尺度数）
bash jobs/submit.sh train bertB 600 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml \
    DATA_NAME=wikitext103_bert TOKENIZER=bert-base-uncased \
    EXTRA_ARGS="--set quantizer.scales=[8,16,32,64,128,256] --set quantizer.revival.interval=150"
# 一台 8×H100 跑 4 个单卡实验的打包方式见 jobs/extra4_entry.sh
```

## 第三轮：冻结前的三大验证（need_next3.md）+ prompt 补测

| 实验 | 干了什么 | 一句话结论 | 结果文件 |
|---|---|---|---|
| Exp 1 `hybrid` | 训 [1,8,16,32,64,128,256]，验证"主题锚+密梯子"两全 | 与 bertB 持平但坡道低 1.4–2pp；q1 锚被稀释（lift 22.6× vs 103.8×） | `results/eval_test_hybrid.json`、`results/probe_planner_hybrid.json` |
| Exp 2 留一尺度 | 每次藏掉一层码看重建掉多少（用 readout decoder 修分布外失真） | **没有冗余尺度**：藏任何一层都掉 0.8–18.2pp | `results/scale_marginal_contribution_{bertB,hybrid}.json` |
| Exp 3 严格探针 | 给全部粗码，预测下一层码，与"什么都不给"的同架构对照比 | **粗码对细码预测零帮助**（增益 −0.46～+0.09 bits）；q1 消融=0 | `results/next_scale_probe_{bertB,hybrid}.json` |
| prompt 补测 | 把前一窗原文也给预测器，重测上面这件事 | **有 prompt 也没用**——结论坐实 | `results/next_scale_probe_prompted_bertB.json` |

```bash
# Exp 2（readout 微调 + 子集评估）/ Exp 3 / prompt 补测 / q1 语义锚探针
bash jobs/submit.sh exp2 bertB 180 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
bash jobs/submit.sh nsp  bertB 180 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
bash jobs/submit.sh nspp bertB 120 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
bash jobs/submit.sh probe bertB 120 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
```

三轮汇总结论见 `docs/reports/` 两份报告和 `results/hybrid_schedule_summary.md`（冻结判定）。

## 版本标签

| tag | 内容 |
|---|---|
| `v0.1-pilot` | 第一轮：基础实现 + base/sd05 |
| `v0.2-bert-line` | 第二轮：BERT 词表 + schedule/因子扫描 |
| `v0.3-next3` | 第三轮：hybrid + 冗余检验 + 严格探针 + 冻结判定 |
