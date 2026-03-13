### 📊 数据集字段定义 (Dataset Schema)

| Column Name | 译义 | 数据类型 | 处理方式 |
| :--- | :--- | :--- | :--- |
| **`user_id`** | 用户 ID | `int` | **删除**：随机噪声，不参与模型计算 |
| **`age`** | 年龄 | `int` | **标准化**：基础人口特征 |
| **`gender`** | 性别 | `str` | **独热编码**：转换为 `0/1` 数值 |
| **`occupation`** | 职业 | `str` | **独热编码**：模型预测的关键维度 |
| **`daily_screen_time_hours`** | 每日屏幕时长 (h) | `float` | **标准化**：核心疲劳诱因 |
| **`phone_usage_before_sleep_minutes`** | 睡前手机时长 (min) | `int` | **标准化**：影响睡眠的重要指标 |
| **`sleep_duration_hours`** | 睡眠时长 (h) | `float` | **标准化**：恢复性指标 |
| **`sleep_quality_score`** | 睡眠质量评分 (1-10) | `int` | **标准化**：主观睡眠质量评估 |
| **`caffeine_intake_cups`** | 咖啡因摄入 (杯) | `int` | **标准化**：生理刺激变量 |
| **`physical_activity_minutes`** | 体育活动 (min) | `int` | **标准化**：压力缓解变量 |
| **`notifications_received_per_day`** | 每日通知数 | `int` | **标准化**：外部干扰频率 |
| **`mental_fatigue_score`** | **心理疲劳分值** | `float` | **标签 1**：线性回归预测目标 |
| **`stress_level`** | **压力等级** | `int` | **标签 2**：Softmax 回归预测目标 |
