import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Hybrid Model (SVM + NN)', 'Combined Model (LR + SVM + NN + RF + XGB)']
lines = ['1000 lines', '10,000 lines', '100,000 lines', 'The entire database']
accuracies = np.array([
    [0.8, 0.85, 0.89, np.nan],  # Hybrid Model
    [0.82, 0.86, 0.9, np.nan]   # Combined Model
])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    ax.plot(lines, accuracies[i], label=method, marker='o')

ax.set_xlabel('Number of Lines')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Hybrid and Combined Models')
ax.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
