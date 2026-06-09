from hsvAlgorithm import getResults

inputArray, goodArray, mediocreArray, poorArray = getResults()

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
labels = ["Input", "Good Ref", "Mediocre Ref", "Poor Ref"]

for ax, arr, label in zip(axes, [inputArray, goodArray, mediocreArray, poorArray], labels):
    ax.imshow(arr, cmap="hot", aspect="auto")
    ax.set_title(label)
    ax.set_xlabel("Hues")
    ax.set_ylabel("Sat.s")

plt.tight_layout()
plt.show()