# 实验 2：层级的最优形状（schedule 扫描）

**问题**：尺度串怎么排最好？密梯子 vs 稀梯子 vs 带超粗端？
**代码**：无专属代码——`train_vqvae.py` + `configs/vqvae_wikitext_bert.yaml`，改 `quantizer.scales`（`revival.interval` 按 25×尺度数配套调整）。

跑过的 5 组：bertPilot [1,2,4,256]、bertD [1,4,16,64,256]、bertA [4..256]、bertB [8..256]、bertC [1..256]、以及第三轮补的 hybrid [1,8..256]。

```bash
bash jobs/submit.sh train bertB 600 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml \
    DATA_NAME=wikitext103_bert TOKENIZER=bert-base-uncased \
    EXTRA_ARGS="--set quantizer.scales=[8,16,32,64,128,256] --set quantizer.revival.interval=150"
```

**结果**：`results/eval_test_bert{Pilot,A,B,C,D}.json`、`results/eval_test_hybrid.json`；跨组对比表 `results_summary.md`（用 `experiments/analyze_runs.py` 生成）。
**结论**：bertB [8..256] 在所有码数预算点最优；超粗尺度 1/2/4 是低效台阶。
