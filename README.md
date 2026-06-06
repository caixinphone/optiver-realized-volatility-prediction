# Optiver - Realized Volatility Prediction (QR 作品集 #2)

> 第二个 QR 作品集，和 [drw-crypto](../drw-crypto/) 互补：DRW 是**加密资产收益预测**，
> 这个是**股票已实现波动率预测**——两大核心 QR 任务都覆盖。
> 赛页：https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
> 主办方 **Optiver**（全球顶级做市商）。

## 一、任务

给定某只股票某个 10 分钟窗口内的**订单簿 + 逐笔成交**数据，预测**下一个 10 分钟窗口**的
**已实现波动率(realized volatility)**。

- 评测指标：**RMSPE**（均方根百分比误差）= `sqrt(mean(((y−ŷ)/y)^2))`，对小目标值的相对误差敏感。
- 这是做市/期权定价的核心：波动率预测准不准，直接决定报价宽窄与风险敞口。

## 二、数据结构（3 张表用 stock_id + time_id 关联）

| 文件 | 内容 |
|---|---|
| `book_[train/test].parquet` | 订单簿快照，按 stock_id 分区。列：`time_id, seconds_in_bucket, bid_price1/2, ask_price1/2, bid_size1/2, ask_size1/2`（最优两档） |
| `trade_[train/test].parquet` | 逐笔成交。列：`time_id, seconds_in_bucket, price, size, order_count` |
| `train.csv` | `stock_id, time_id, target`（下一窗口已实现波动率） |
| `test.csv` / `sample_submission.csv` | `stock_id, time_id, row_id` / `row_id, target` |

约 112 只股票 × ~3830 个 time_id（time_id 是**乱序匿名**的，不是时间顺序 → 用 KFold/GroupKFold，不用时序 CV）。

## 三、核心公式与特征工程（`features.py`）

**WAP（加权平均价，本赛的"价格"定义）**
```
WAP1 = (bid_price1*ask_size1 + ask_price1*bid_size1) / (bid_size1 + ask_size1)
```
**对数收益 & 已实现波动率（与 target 同定义，是最强特征）**
```
log_return = diff(log(WAP))
realized_vol = sqrt( sum(log_return^2) )
```

每个 (stock_id, time_id) 桶内聚合出特征：
- **订单簿**：WAP1/WAP2 的已实现波动率、买卖价差 `ask1/bid1−1`、两档价差、
  挂单失衡 `(bid_size−ask_size)/(bid_size+ask_size)`、总深度、价格/挂单量的 mean/std/sum。
- **成交**：成交价的已实现波动率、总成交量 `sum(size)`、成交笔数、`order_count` 之和、成交额。
- **子窗口特征(关键)**：只用桶内**后半段**(`seconds_in_bucket ≥ 150/300/450`)重算上述波动率——
  越靠近窗口末尾，越能预测下一窗口，子窗口波动率通常是 Top 特征。
- **跨股票/跨时刻聚合**：按 `stock_id` 的特征均值/方差(类目标编码)；按 `time_id` 跨股票均值
  (捕捉**全市场波动率 regime**)。

## 四、模型与验证（`train.py`）

- **LightGBM**，直接优化 **RMSPE**：用 `sample_weight = 1/target^2` 配 MSE，等价于最小化 RMSPE。
- **KFold(5)**（time_id 乱序，无需时序切分；可选 GroupKFold by time_id 防 time_id 级泄漏）。
- 报告 OOF RMSPE；可加线性/NN 集成。

## 五、怎么跑

```bash
cd "optiver-vol"
pip install -r requirements.txt
# 配置 ~/.kaggle/kaggle.json 或 export KAGGLE_KEY=...，并在赛页接受规则
python download_data.py          # 数据较大(数 GB)
python build_features.py         # 订单簿/成交 → 每 (stock,time) 一行特征，存 parquet
python train.py                  # KFold + LightGBM(RMSPE) → output/submission.csv
# 或一键：bash run_all.sh
```

## 六、简历写法（示例）

> **Optiver Realized Volatility Prediction（Kaggle，Optiver 主办）** — 从股票订单簿/逐笔成交
> 构建已实现波动率预测模型；WAP 对数收益、子窗口已实现波动率、挂单/成交失衡等微观结构特征
> + 跨股票/跨时刻聚合；LightGBM 直接优化 RMSPE，KFold 验证。与 DRW 加密收益预测互补，
> 覆盖"波动率建模 + 收益预测"两类核心 QR 任务。

## 实际结果（2026-06，全量数据）

- **私榜 RMSPE = 0.23619**（真实提交分数；本赛**冠军 ≈ 0.196**，3809 队）。
- 本地 **5 折 CV RMSPE = 0.21623**（各折 0.2131–0.2188，非常稳定）；428,932 行、91 个特征、LightGBM。
- CV 0.216 → 私榜 0.236 的差距 = 正常的 CV 乐观偏差；单模型基线做到 0.236 属扎实中游。

**提交方式（code competition 的正规流程,已全程跑通)**：`test.csv` 只有 3 行占位、真实 test 隐藏，
CSV 无法得到真实分数。改为把整条流水线打包成自包含 Kaggle kernel（[kernel/optiver_submit.py](kernel/optiver_submit.py)）→
`kaggle kernels push` 在线运行 → `kaggle competitions submit -k <kernel> -v <ver> -f submission.csv`，
由 Kaggle 在**完整隐藏 test** 上重跑打分 → 私榜 0.23619。

## 七、与 DRW 的差异（面试谈资）
- DRW 是**扁平表格 + 匿名信号**；这个是**多表、原始订单簿**，要自己从微观结构造特征——更能体现你懂市场。
- DRW 用**时序稳定性**选特征(数据按时间排);这里 time_id **乱序**，改用 KFold——说明你会**按数据性质选验证方案**。
