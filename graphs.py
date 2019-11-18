import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()

x = np.linspace(0, 1, 20).reshape(-1, 1)
y = x

stage0 = np.array([1, 1])
stage1 = np.array([0.02, 1])
stage2 = np.array([0.0005, 1])

ax = fig.add_subplot(111)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("ROC Space")
ax.plot(x,y, c='r', linestyle='--', linewidth=2, label='Random guess')

ax.plot(stage0[0], stage0[1], marker='o', color='b', label='Stage 0')
ax.plot(stage1[0], stage1[1], marker='o', color='g', label='Stage 1')
ax.plot(stage2[0], stage2[1], marker='o', color='y', label='Stage 2')

ax.legend()
fig.show()