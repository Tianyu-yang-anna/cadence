# 实验 5：粗码能预测细码吗？（Stage 1 前提检验，本仓库最重要的实验）

**问题**：VAR planner 的假设是"生成粗码后细码变好猜"——真的吗？
**代码（本文件夹）**：
- `probe_planner.py` —— 泛用探针：tiny AR 测 flatten 码序列可预测性 + q1 相邻窗口一致性（主题指纹检验）。
- `probe_next_scale.py` —— 严格探针：按 VAR 分解（给定全部粗码、并行逐位置预测目标层），对照组=同架构+空条件；含 q256 增量条件与 q1 消融。
- `probe_next_scale_prompted.py` —— 在上面基础上把前一窗原文作为 prompt 加入条件，模拟真实生成场景。

```bash
bash jobs/submit.sh probe bertB 120 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
bash jobs/submit.sh nsp   bertB 180 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
bash jobs/submit.sh nspp  bertB 120 1xh100 RUN_NAME=bertB CONFIG=configs/vqvae_wikitext_bert.yaml DATA_NAME=wikitext103_bert
```

**结果**：`results/next_scale_probe_{bertB,hybrid}.json`、`results/next_scale_probe_prompted_bertB.json`、`results/probe_planner_{bertPilot,hybrid}.json`
**结论**：粗码对细码预测增益 ≈0（有无 prompt 都是）；q1 消融=0。原因是残差量化天生使各层近独立。→ tokenizer 未冻结，Stage 0.5 改进方向见主 README。
**已知限制**：探针以码 id 为输入（4L×256d）；用累计反量化 latent 作输入的向量版探针是待办。
