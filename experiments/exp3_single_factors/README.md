# 实验 3：单因子消融（φ conv / 独立码本 / dropout 0.75 / 词表）

**问题**：这几个设计旋钮各值多少分？
**代码**：无专属代码——都是 `train_vqvae.py` + `--set` 单因子改动（基线 bertPilot）。

```bash
# φ conv
... EXTRA_ARGS="--set quantizer.phi.enabled=true"
# 每尺度独立码本
... EXTRA_ARGS="--set quantizer.shared_codebook=false"
# 更强 dropout
... EXTRA_ARGS="--set train.scale_dropout_p=0.75"
```
（四个实验打包在一台 8×H100 上并行跑的方式见 `jobs/extra4_entry.sh`）

**结果**：`results/eval_test_bert{Phi,SepCB,P75}.json`
**结论**：φ 无效弃用；独立码本 +0.3pp 边际；dropout 0.75≈0.5（瓶颈是容量不是压力）；BERT vs GPT-2 词表差异 <0.1pp。
