from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
result_dir = root / "kalman" / "results"

filter_path = result_dir / "stock_filter_result.csv"
predict_path = result_dir / "stock_predict_result.csv"

history = np.genfromtxt(filter_path, delimiter=",", names=True)
future = np.genfromtxt(predict_path, delimiter=",", names=True)

day = history["day"]
original = history["original"]
kalman = history["kalman"]

future_day = future["day"]
future_price = future["predict"]

plt.figure(figsize=(10, 6))

# 原始股票价格：用散点表示
plt.scatter(
    day,
    original,
    color="gray",
    s=18,
    alpha=0.65,
    label="Original Price"
)

# 卡尔曼滤波后的价格：蓝色曲线
plt.plot(
    day,
    kalman,
    color="blue",
    linewidth=2,
    label="Kalman Filtered Price"
)

# 未来预测价格：红色虚线 + 圆点
plt.plot(
    future_day,
    future_price,
    color="red",
    linestyle="--",
    marker="o",
    markersize=5,
    linewidth=2,
    label="Future Prediction"
)

plt.title("Stock Price Prediction with Kalman Filter")
plt.xlabel("Day")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.legend()

output_path = result_dir / "stock_prediction.png"
plt.savefig(output_path, dpi=200)
plt.close()

print("股票预测图已保存:", output_path)