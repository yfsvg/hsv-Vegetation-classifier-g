from hsvAlgorithm import getResults

inputArray, canopyArray, lowShrubArray, patchyArray, dryArray = getResults()

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
labels = ["Input", "Canopy ref", "Low shrub ref", "Patchy ref", "Dry ref"]

for ax, arr, label in zip(axes, [inputArray, canopyArray, lowShrubArray, patchyArray, dryArray], labels):
    ax.imshow(arr, cmap="hot", aspect="auto")
    ax.set_title(label)
    ax.set_xlabel("Hues")
    ax.set_ylabel("Sat.s")

plt.tight_layout()
plt.show()