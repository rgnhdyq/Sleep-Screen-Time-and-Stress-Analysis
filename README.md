# Mental Health Prediction with PyTorch 🧠

![Status](https://img.shields.io/badge/Status-Completed-success)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-00BFFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/jayjoshi37/sleep-screen-time-and-stress-analysis/data)

本项目基于 PyTorch 框架，对职场人员及学生的睡眠、屏幕时长与压力数据进行深度分析。通过构建双重预测模型，实现了对**心理疲劳值**的连续预测及**压力等级**的离散分类。

---

## 📊 数据集定义 (Dataset Schema)

数据集来源：[Sleep, Screen Time, and Stress Analysis](https://www.kaggle.com/datasets/jayjoshi37/sleep-screen-time-and-stress-analysis/data)

| 特征名称 | 译义 | 数据类型 | 预处理策略 |
| :--- | :--- | :--- | :--- |
| `user_id` | 用户 ID | `int` | **删除**：随机噪声，不参与模型计算 |
| `age` | 年龄 | `int` | **标准化**：基础人口特征 |
| `gender` | 性别 | `str` | **One-Hot**：转换为 `0/1` 数值 |
| `occupation` | 职业 | `str` | **One-Hot**：捕捉不同行业压力特征 |
| `daily_screen_time_hours` | 每日屏幕时长 (h) | `float` | **标准化**：核心疲劳诱因 |
| `phone_usage_before_sleep_min` | 睡前手机时长 (min) | `int` | **标准化**：睡眠干扰因素 |
| `sleep_duration_hours` | 睡眠时长 (h) | `float` | **标准化**：身心恢复指标 |
| `sleep_quality_score` | 睡眠质量评分 | `int` | **标准化**：主观感受量化 |
| `caffeine_intake_cups` | 咖啡因摄入 | `int` | **标准化**：生理刺激变量 |
| `physical_activity_minutes` | 体育活动时长 | `int` | **标准化**：压力缓解变量 |
| `notifications_received` | 每日通知数 | `int` | **标准化**：外部信息干扰 |
| **`mental_fatigue_score`** | **心理疲劳分值** | `float` | **Label 1**：线性回归目标 (Regression) |
| **`stress_level`** | **压力等级** | `float` | **Label 2**：Softmax 分类目标 (Classification) |

---

## 🚀 项目亮点

- **双模并行分析**：
  - **线性回归**：精准预测疲劳分数 ($0.0 \sim 10.0$)。
  - **Softmax 回归**：通过分桶预处理，将压力识别为低、中、高三个等级。
- **自动化可视化**：训练脚本自动生成并导出 Loss 收敛曲线及模型权重。

---

## 🛠️ 环境安装

```bash
# 安装项目依赖
pip install -r requirements.txt
