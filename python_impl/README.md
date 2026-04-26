# Python 重写说明

这套代码把你论文里 Chapter 4 的核心实验按函数式方式重写成了 Python，不依赖复杂 class。

## 对应关系

- `python_impl/experiments/exp_4_1_gaussian_speed_accuracy.py`
  - 对应论文 4.1
  - 比较 Gaussian target 下 MCMC 与 FGVI 的均值/协方差估计精度
- `python_impl/experiments/exp_4_2_logistic_regression.py`
  - 对应论文 4.2.1
  - Pima 和 German 两个 logistic regression 后验实验
- `python_impl/experiments/exp_4_2_gaussian_hmc_gvi.py`
  - 对应论文 4.2.2
  - 100 维 Gaussian target 的 HMC / HMC-GVI 实验
- `python_impl/experiments/exp_4_2_glmm.py`
  - 对应论文 4.2.3
  - Polypharm 数据上的 509 维 GLMM 实验

## 核心模块

- `python_impl/data.py`
  - 读取 Pima、German、Polypharm 数据
- `python_impl/targets.py`
  - 定义 Gaussian、logistic posterior、GLMM posterior 及其梯度
- `python_impl/vi.py`
  - CGVI
  - Gaussian target 下的 FGVI
  - Polypharm GLMM 的 sparse-precision GVI
- `python_impl/mcmc.py`
  - RMH
  - AM
  - MALA
  - HMC
- `python_impl/metrics.py`
  - `efficiency`
  - `lag1 autocorrelation`
  - Gaussian moment RMSE

## 运行方式

在项目根目录运行：

```bash
python -m python_impl.experiments.exp_4_1_gaussian_speed_accuracy --mode quick
python -m python_impl.experiments.exp_4_2_logistic_regression --mode quick
python -m python_impl.experiments.exp_4_2_gaussian_hmc_gvi --mode quick
python -m python_impl.experiments.exp_4_2_glmm --mode quick
```

如果你想尽量贴近论文原始设定，可以把 `--mode quick` 改成 `--mode paper`。但 `paper` 模式会非常耗时，尤其是 GLMM。

## 输出

每个实验脚本都会：

- 在终端打印结果表
- 把结果保存成 CSV

默认输出文件：

- `python_impl/results_exp_4_1.csv`
- `python_impl/results_exp_4_2_logistic.csv`
- `python_impl/results_exp_4_2_gaussian.csv`
- `python_impl/results_exp_4_2_glmm.csv`

## 备份

当前目录已经生成代码和数据备份包：

- `HMC-GVI_code_data_backup_before_python_refactor.zip`

原 PDF 文件因为当时被其他进程占用，没有一起压进去；如果你关掉 PDF，我可以再补一个包含论文文件的完整备份包。
