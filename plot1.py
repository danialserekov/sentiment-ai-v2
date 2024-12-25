import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Logistic Regression (LR)', 'Support Vector Machine (SVM)', 'Neural Network (NN)', 'Random Forest (RF)', 'XGBoost (XGB)']
lines = ['1000 lines', '10,000 lines', '100,000 lines', 'The entire database']
accuracies = np.array([
    [0.78, 0.83, 0.85, 0.87],  # LR
    [0.79, 0.84, 0.90, np.nan],  # SVM
    [0.83, 0.85, 0.89, np.nan],  # NN
    [0.80, 0.82, 0.84, 0.87],  # RF
    [0.82, 0.83, 0.85, 0.88]   # XGB
])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    ax.plot(lines, accuracies[i], label=method, marker='o')

ax.set_xlabel('Number of Lines')
ax.set_ylabel('Accuracy')
ax.set_title('Algorithm Accuracy vs. Number of Lines')
ax.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
