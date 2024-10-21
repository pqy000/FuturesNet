import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data = np.load('../newdata/50_2020.npy')

# 提取各个维度的数据
open_prices = data[:, 0]
close_prices = data[:, 1]
high_prices = data[:, 2]
low_prices = data[:, 3]
volume = data[:, 4]
trend = data[:, 5]

# 1. 统计分析
open_mean = np.mean(open_prices)
close_mean = np.mean(close_prices)
high_mean = np.mean(high_prices)
low_mean = np.mean(low_prices)
volume_mean = np.mean(volume)

# 打印统计分析结果
print(f"开盘价平均值: {open_mean:.2f}")
print(f"收盘价平均值: {close_mean:.2f}")
print(f"最高价平均值: {high_mean:.2f}")
print(f"最低价平均值: {low_mean:.2f}")
print(f"成交量平均值: {volume_mean:.2f}")

# 2. 价格趋势的可视化
plt.figure(figsize=(10, 6))
plt.plot(open_prices[:1000], label='Open Price', alpha=0.7)
plt.plot(close_prices[:1000], label='Close Price', alpha=0.7)
plt.plot(high_prices[:1000], label='High Price', alpha=0.7)
plt.plot(low_prices[:1000], label='Low Price', alpha=0.7)
plt.title('Price Trends (First 1000 Minutes)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Price')
plt.legend()
plt.show()

# 3. 成交量的变化趋势
plt.figure(figsize=(10, 6))
plt.plot(volume[:1000], label='Volume', color='orange', alpha=0.7)
plt.title('Volume Trends (First 1000 Minutes)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Volume')
plt.legend()
plt.show()

# 4. 期货涨跌趋势分布
plt.figure(figsize=(8, 5))
plt.hist(trend, bins=6, edgecolor='black', color='purple', alpha=0.7)
plt.title('Distribution of Trend Labels')
plt.xlabel('Trend')
plt.ylabel('Frequency')
plt.show()