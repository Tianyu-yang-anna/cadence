# 作业提交记录（tracked 副本）

`jobs/submit.sh` 在仓库根生成的 `.job-cadence-*.yaml` 被 gitignore（生成物），
导致"某个臂到底用什么超参训的"只存在于本地——审计与复核多次被它卡住。本目录是
这些提交记录的 tracked 副本（去掉文件名前导点，避开 gitignore 的 `.job-*.yaml` 规则）：**每个臂/评测作业的 env_variables 就是它的完整配置**
（EXTRA_ARGS 含训练旗标，SAMPLE_MODE 含解码超参）。

同名文件 = 同一 experiment_name 的最近一次提交（重投会覆盖）。与
`tools/build_master_table.py` 的 ROWS 登记表互为印证。
