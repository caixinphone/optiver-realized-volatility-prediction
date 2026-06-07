# Optiver — Realized Volatility Prediction

> QR 作品集：从订单簿 + 逐笔成交预测股票**下一 10 分钟的已实现波动率**（做市与期权定价的核心）。
> 主办方 **Optiver**（全球顶级做市商）。指标 **RMSPE**。赛页：
> https://www.kaggle.com/competitions/optiver-realized-volatility-prediction

**这个项目真正的看点不是名次，而是一次教科书级的"泄漏诊断"**：朴素 KFold 给出 0.192 的
诱人分数，但严谨的 GroupKFold 揭穿它是 0.213 的假象——私榜 0.216 最终验证了 GroupKFold。
这正是量化回测里"看起来能赚钱、实盘亏钱"的同一类陷阱。

---

## 一、结果总览（私榜 = 隐藏 test 真实分数）

| 版本 | 方法 | 本地 CV | **私榜 RMSPE** |
|---|---|---|---|
| V1 | 订单簿微观结构 + LightGBM | KFold 0.2162 | 0.23619 |
| V2 | + 横截面聚合 + 最近邻特征；LGB(250树)+CatBoost | GroupKFold 0.2135 | 0.21627 |
| **V2-blend** | **+ MLP(stock embedding) 三模型融合** | GroupKFold **0.2133** | **0.21544** |

- 冠军 ≈ 0.196，3852 队。**0.2154 ≈ top 12–15%（铜牌附近）**，较 V1 降 **8.8%**。
- 私榜 0.215 与 GroupKFold 0.213 高度吻合 → 验证了诚实 CV 的判断。
- 三模型融合较单 GBM 私榜降 0.0008（NN 与树模型误差结构互补，诚实增益）。

## 二、核心方法学：为什么 KFold 在本赛会骗人

time_id 是**乱序匿名**的，私榜是**全新的 time_id**。但有些特征（按 time_id 跨股票聚合的
"市场 regime"、最近邻聚合）会让模型在随机 KFold 下**背诵每个 time_id 的全市场波动水平**——
同一个 time_id 的其它股票就躺在训练折里。这在私榜（没见过的 time_id）上无法复现。

| 模型 | KFold(乱序) | GroupKFold(by time_id) |
|---|---|---|
| LightGBM(V2 特征) | 0.1925 | **0.2135** |

裂口 **0.021**。铁证：LGB 在 GroupKFold 下早停最优树数只有 ~200（KFold 下能到上千）——多出来的
树都在背 time_id。**所以最终模型只用 250 棵树**（V1 当年用 1400 棵正是这个过拟合，私榜才 0.236）。
**私榜 0.216 ≈ GroupKFold 0.213，而非 KFold 0.192 → GroupKFold 才是本赛诚实的回测口径。**

## 三、特征工程（`features_v2.py` / `knn_features.py`）

- **桶内**：WAP/WAP1/WAP2 的已实现波动率 + realized quarticity；两档买卖价差；带符号挂单失衡；
  成交 `tau`、平均成交笔大小；子窗口(seconds≥150/300/450)版本。对数收益向量化(`np.log` + groupby diff)。
- **横截面聚合**：对关键特征按 `stock_id`(个股典型水平) 与 `time_id`(全市场 regime) 各取 mean/std。
  **全程只聚合输入特征、绝不碰 target → 无折泄漏**，且与隐藏 test 上的算法完全一致。
- **最近邻聚合**（本赛榜一 "Nearest Neighbors" 核心）：把 `[time_id × stock]` 波动率矩阵分别在
  time 轴与 stock 轴找最近邻，聚合波动率/成交量等（denoise + 注入同 regime 同业信息）。

## 四、模型（`train_v2.py` / `nn_model.py`）

- **LightGBM**：`sample_weight=1/y²` 配 RMSE 目标 = 直接优化 RMSPE。250 棵树（GroupKFold 定）。
- **CatBoost**：同 RMSPE 加权，提供 GBM 内部多样性。
- **MLP + stock embedding**：直接以 RMSPE 为损失。**关键工程坑**：微观结构特征重尾偏态，
  NN 必须先 `log1p` 再标准化（否则 RMSPE 0.33；修复后 0.215，与 LGB 同档）；梯度裁剪防偶发发散；
  部署用 3-seed 平均。
- **融合**：在 **GroupKFold OOF** 上搜最优权重(≈ lgb .74 / cat .16 / nn .10)，用诚实口径选权重。

## 五、冲金（≤0.20）的诚实结论

试过榜一的全部公开手法仍卡在 GroupKFold 0.213：rank 变换(对 LGB 无效)、更厚的 KNN(无效)、
**stassl 的 time_id 时序还原**（`recover_order.py`：对平均价格矩阵做 t-SNE 1D 还原顺序 + AR 波动率特征）。
诊断发现：**还原只到 regime 级**——沿还原序的市场波动率 lag-1 自相关 0.31(随机仅 0.02)，但 lag-1..10
全平、不衰减，说明 t-SNE 还原的是"相似波动时段的聚类"而非逐窗口真实先后。根因：数据**按窗口独立
归一化**(每窗口起始价≈1.0)，价格流形给得出 regime、给不出真时序，而 AR 信号恰需后者。
**结论**：冲金需高保真复刻榜一全流程(精还原+海量特征+精调 NN/CNN/GBM 集成)，是数周研究量。

## 六、怎么跑

```bash
cd optiver-vol
pip install -r requirements.txt            # lightgbm catboost torch scikit-learn pandas pyarrow
python download_data.py                    # 需 ~/.kaggle/kaggle.json 且已在赛页接受规则(~1.6GB)
python build_features_v2.py                # 桶内+横截面+KNN → output/train_features_v2.parquet
python train_v2.py --cv group --models both        # 诚实 GroupKFold:LGB+CatBoost
python train_nn_v2.py --cv group --epochs 70       # MLP(log变换) GroupKFold
python blend.py --cv group                         # 三模型诚实最优权重
```

提交（Optiver 是 code competition，test.csv 仅 3 行占位、真实 test 隐藏）：自包含 kernel
`kernel_v2/optiver_submit_v2.py` → `kaggle kernels push -p kernel_v2` 在线重跑 → `kaggle competitions
submit -k caixin030703/optiver-rv-v2-blend -v <版本> -f submission.csv`，由 Kaggle 在完整隐藏 test 打分。

**踩坑**：① 竞赛数据挂在 `/kaggle/input/competitions/<slug>/`，用 glob 自动探测。② Kaggle 默认 GPU 是
P100(sm_60)，预装 PyTorch 只支持 sm_70+ → torch 在 P100 直接报错；NN 改跑 CPU（CatBoost 仍可用 P100）。
③ `kaggle kernels status` 接口偶发 500，改轮询输出日志判断完成。
