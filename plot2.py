import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Hybrid Model (SVM + NN)', 'Combined Model (LR + SVM + NN + RF + XGB)']
lines = ['1000 lines', '10,000 lines', '100,000 lines', 'The entire database']
times = np.array([
    [10.62, 208.32, 3102.22, 4.2 * 3600],  # Hybrid Model
    [10.05, 427.45, 5465.37, 8.3 * 3600]   # Combined Model
])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    ax.plot(lines, times[i], label=method, marker='o')

ax.set_xlabel('Number of Lines')
ax.set_ylabel('Time (seconds)')
ax.set_title('Execution Time of Hybrid and Combined Models')
ax.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
