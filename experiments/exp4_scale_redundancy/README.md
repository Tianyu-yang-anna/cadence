# 实验 4：尺度冗余检验（leave-one-scale-out）

**问题**：每层码是独特信息还是和邻居重复？
**代码（本文件夹）**：
- `finetune_subset_readout.py` —— 冻结 encoder+码本，把 decoder 复制一份在随机尺度子集上微调 2k 步。原 decoder 只见过"前缀"输入，直接藏码评估是分布外（会把粗尺度损失夸大 ~2 倍），这个 readout 是可信读数的前提。
- `eval_scale_subsets.py` —— 自动生成条件（全量 / 留一尺度 / 单尺度 / 邻居组合），raw 与 readout 两种模式各出一张表。

```bash
bash jobs/submit.sh exp2 bertB 180 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
```

**结果**：`results/scale_marginal_contribution_{bertB,hybrid}.json`
**结论**：没有冗余尺度——藏掉任一层掉 0.8~18.2pp、随该层码数单调；邻居互补不重叠。
