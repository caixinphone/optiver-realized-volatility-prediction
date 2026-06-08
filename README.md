# Optiver — Realized Volatility Prediction

从**订单簿快照 + 逐笔成交**预测股票**未来 10 分钟的已实现波动率**——做市报价与期权定价的核心问题。
主办方 [Optiver](https://optiver.com)（全球顶级做市商），Kaggle Featured 赛，指标 **RMSPE**，3852 支队伍。

> **本项目的看点不止于名次，而是一次完整的"防过拟合"实战**：朴素 KFold 给出 0.192 的诱人分数，
> 但严格的 GroupKFold 揭穿它是 **0.213** 的假象——私榜最终 **0.215** 验证了后者。这正是量化回测中
> "曲线很美、实盘亏钱"的同一类陷阱。识别并量化它，是这份作品想展示的核心能力。

---

## 成绩

| 迭代 | 方法 | 诚实 CV(GroupKFold) | **私榜 RMSPE** |
|:--|:--|:--:|:--:|
| 基线 | 订单簿微观结构特征 + LightGBM | — | 0.23619 |
| 特征升级 | + 横截面聚合 + 最近邻聚合；LGB + CatBoost | 0.2135 | 0.21627 |
| **最终** | **+ MLP(stock embedding) 三模型融合** | **0.2133** | **0.21544** |

私榜 **0.21544**，较基线降 **8.8%**，约处 **top 12–15%**。（冠军 ≈ 0.196。）

## 核心方法学：为什么本赛 KFold 会骗人

`time_id` 是**乱序匿名**的，私榜评测用的是一批**全新的 time_id**。但部分特征（按 time_id 跨股票聚合的
"全市场波动 regime"、最近邻聚合）会让模型在随机 KFold 下**记住每个 time_id 的波动水平**——因为同一个
time_id 的其它股票就躺在训练折里。这种"记忆"在私榜（没见过的 time_id）上无法复现。

| 模型 | KFold(随机) | GroupKFold(按 time_id) |
|:--|:--:|:--:|
| LightGBM | 0.1925 | **0.2135** |

裂口高达 **0.021**。铁证：GroupKFold 下 LightGBM 的早停最优树数只有 **~200**（随机 KFold 下能到上千）——
多出来的树都在背诵 time_id。**所以最终模型只用 250 棵树**（基线用 1400 棵正是过拟合，私榜才 0.236）。
**私榜 0.215 ≈ GroupKFold 0.213 ≠ KFold 0.192 → GroupKFold 才是本赛诚实的回测口径。**

## 方法

**特征工程**（[features.py](features.py) / [knn_features.py](knn_features.py)）
- *桶内*：WAP/WAP1/WAP2 的已实现波动率 + realized quarticity；两档买卖价差；带符号挂单失衡；
  成交 `tau`、平均成交笔大小；按 `seconds_in_bucket ≥ 150/300/450` 的子窗口版本（越靠窗口末端越能预测下一窗口）。
- *横截面聚合*：对关键特征按 `stock_id`（个股典型水平）与 `time_id`（全市场 regime）取 mean/std。
  **全程只聚合输入特征、绝不使用 target → 无折间泄漏**，且与隐藏 test 上的算法完全一致。
- *最近邻聚合*：把 `[time_id × stock]` 波动率矩阵分别在 time 轴与 stock 轴找最近邻并聚合
  （denoise + 注入"同 regime、同业股票"的信息）。

**模型**（[train.py](train.py) / [nn_model.py](nn_model.py) / [blend.py](blend.py)）
- **LightGBM** & **CatBoost**：以 `sample_weight = 1/y²` 配 RMSE 目标，等价于直接优化 RMSPE。
- **MLP + stock embedding**：直接以 RMSPE 为损失。关键工程点——微观结构特征**重尾偏态**，
  NN 必须先 `log1p` 再标准化（否则 RMSPE 0.33，修复后 0.215，与 GBM 同档），配合梯度裁剪与 3-seed 平均防发散。
- **融合**：在 **GroupKFold OOF** 上搜最优权重（≈ lgb .74 / cat .16 / nn .10），即用诚实口径选权重；
  NN 与树模型误差结构互补，给私榜带来 0.0008 的诚实增益。

## 探索的边界（冲击金牌区的尝试与结论）

为突破 GroupKFold 0.213 的平台期，复现了 1st-place "Nearest Neighbors" 与 stassl 的 **time_id 时序还原**
（对 `[time_id × stock]` 平均价格矩阵做 t-SNE 1D 还原顺序）。**诊断结论**：还原确实有效但只到 *regime 级*——
沿还原序的市场波动率 lag-1 自相关 0.31（随机序仅 0.02），但 lag-1…10 全平、不衰减，说明它聚的是
"相似波动时段"而非"逐窗口真实先后"。根因：数据**按窗口独立归一化**（每窗口起始价 ≈ 1.0），价格流形
给得出 regime、给不出真时序，而自回归波动率信号恰需后者。冲金需高保真复刻冠军全流程，属数周级研究量。

## 项目结构

```text
.
├── download_data.py        # 下载竞赛数据（需 kaggle.json，~1.6GB）
├── features.py             # 桶内特征 + 横截面聚合
├── knn_features.py         # 最近邻聚合特征
├── build_features.py       # 编排：原始 parquet → 特征表
├── train.py                # LightGBM + CatBoost，KFold / GroupKFold
├── train_nn.py             # MLP(stock embedding) 的 CV
├── nn_model.py             # MLP 定义 + 训练（log 预处理、RMSPE 损失）
├── blend.py                # 三模型 OOF 最优权重搜索
├── kernel/                 # 自包含提交 kernel（Kaggle code competition）
└── requirements.txt
```

## 复现

```bash
pip install -r requirements.txt
python download_data.py                       # 需 ~/.kaggle/kaggle.json 且已在赛页接受规则
python build_features.py                       # → output/train_features.parquet
python train.py     --cv group --models both   # 诚实 GroupKFold：LGB + CatBoost
python train_nn.py  --cv group --epochs 70     # MLP（log 变换）
python blend.py     --cv group                 # 三模型诚实最优权重
```

**提交**（Optiver 是 code competition，`test.csv` 仅 3 行占位、真实 test 隐藏）：把整条流水线打包成自包含
kernel（[kernel/optiver_submit.py](kernel/optiver_submit.py)），`kaggle kernels push -p kernel` 在线重跑，
再 `kaggle competitions submit -k <kernel> -v <版本> -f submission.csv`，由 Kaggle 在完整隐藏 test 上打分。

## 工程踩坑

- 竞赛数据挂载在 `/kaggle/input/competitions/<slug>/`（非 `/kaggle/input/<slug>/`），用 glob 自动探测。
- Kaggle 默认 GPU 是 **Tesla P100（sm_60）**，预装 PyTorch 只支持 sm_70+，torch 在其上直接报错；
  故 NN 在 kernel 内**强制 CPU**，CatBoost 仍可用 GPU。
- `kaggle kernels status` 接口偶发 500，改为轮询输出日志判断完成。
