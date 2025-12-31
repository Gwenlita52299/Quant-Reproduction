1. 使用以下命令创建动量转换器输入特征：`python -m Process.create_feature`。

2. （可选）运行变化点检测模块：`python -m Process.cpd_all <<CPD_WINDOW_LENGTH>>`，例如 `python -m Process.cpd_all 21` 和 `python -m Process.cpd_all 126`。

3. 在变化点检测模块完成后，使用以下命令创建动量转换器输入特征，包括 CPD 模块特征：`python -m Process.create_feature 21`。

4. 要创建包含多个变化点检测回溯窗口的特征文件，请在 126 天 LBW 变化点检测模块完成后，且 21 天 LBW 的特征文件存在后，运行以下命令：`python -m Process.create_feature 126 21`。

5. 运行 Momentum Transformer 或 Slow Momentum with Fast Reversion 实验，运行以下命令：`python -m Process.run_experiment <<EXPERIMENT_NAME>>`
