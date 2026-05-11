from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
result_dir = root / "kalman" / "results"

files = [
    ("homework_data_1", "File 1", 1),
    ("homework_data_2", "File 2", 2),
    ("homework_data_3", "File 3", 1),
    ("homework_data_4", "File 4", 5),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

equation_file = open(result_dir / "fitting_equations.txt", "w", encoding="utf-8")

for ax, item in zip(axes, files):
    name, title, degree = item
    csv_path = result_dir / f"{name}_result.csv"

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    x = data["x"]
    original = data["original"]
    kalman = data["kalman"]

    ax.scatter(x, original, s=2, color="blue", alpha=0.12, label="Original")
    ax.plot(x, kalman, color="red", linewidth=2, label="Kalman Filtered")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    coeffs = np.polyfit(x, kalman, degree)
    poly = np.poly1d(coeffs)

    equation_file.write(f"{title} / {name}\n")
    equation_file.write(f"{degree}阶拟合方程:\n")
    equation_file.write(str(poly))
    equation_file.write("\n\n")

equation_file.close()

plt.tight_layout()
plt.savefig(result_dir / "homework1_kalman_result.png", dpi=200)
plt.close()

print("图片已保存:", result_dir / "homework1_kalman_result.png")
print("拟合方程已保存:", result_dir / "fitting_equations.txt")