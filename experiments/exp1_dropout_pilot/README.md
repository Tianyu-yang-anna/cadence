# 实验 1：层级会自然涌现吗？（scale dropout 对照）

**问题**：不加任何压力，粗尺度会自己学到信息吗？
**代码**：无专属代码——就是根目录 `train_vqvae.py` + `configs/vqvae_wikitext.yaml`，唯一变量是 `train.scale_dropout_p`（0 vs 0.5）。

```bash
bash jobs/submit.sh train base 480 1xh100 RUN_NAME=base CONFIG=configs/vqvae_wikitext.yaml
bash jobs/submit.sh train sd05 480 1xh100 RUN_NAME=sd05 CONFIG=configs/vqvae_wikitext.yaml \
    EXTRA_ARGS="--set train.scale_dropout_p=0.5"
```

**结果**：`results/eval_test_{base,sd05}.json`
**结论**：不会自然涌现（base 粗码解码 0.9%、decoder 对截断输入分布外）；dropout 0.5 逼出层级（6.7%）且全量重建只降 0.03pp。详见主 README 实验 1。
