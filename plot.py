import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Logistic Regression (LR)', 'Support Vector Machine (SVM)', 'Neural Network (NN)', 'Random Forest (RF)', 'XGBoost (XGB)']
lines = ['1000 lines', '10,000 lines', '100,000 lines', 'The entire database']
times = np.array([
    [2.49, 1.92, 25.11, 328.39],  # LR
    [0.28, 17.03, 1035.19, np.nan],  # SVM
    [15.91, 138.38, 1210.55, np.nan],  # NN
    [1.35, 20.43, 309.17, 4678.71],  # RF
    [2.11, 16.26, 125.30, 965.37]   # XGB
])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    ax.plot(lines, times[i], label=method, marker='o')

ax.set_xlabel('Number of Lines')
ax.set_ylabel('Time (seconds)')
ax.set_title('Algorithm Execution Time vs. Number of Lines')
ax.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
