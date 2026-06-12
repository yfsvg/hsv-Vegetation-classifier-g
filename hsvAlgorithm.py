from PIL import Image
import cv2
import numpy as np
import heapq

# Image that is to be analyzed
IMAGE_PATH = "referenceImages/INPUT3.png"

CANOPY_REF = "referenceImages/canopy.png"
LOWSHRUB_REF = "referenceImages/lowShrub.png"
PATCHY_REF = "referenceImages/patchy.png"
DRY_REF = "referenceImages/dry.png"

# 12 buckets, each containing a different grouping of colors, with steps of 14.
# these are all maximum HSV values, so for example "0" represents 0-14 range. "120" is 120-134.
HUE_Buckets = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168];

# 8 categorizes for saturation. each step is 12.5.
SAT_Buckets = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5]

def valueNoise(imagePath: str) -> float:
    img_bgr = cv2.imread(imagePath)
    # Blotting out microchanges
    img_bgr = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = img_hsv[:, :, 2].astype(np.float32)

    # this time ignoring all black or all white (image errors / dead pixels)
    # We are specifically looking for value fluctuations now!
    valid_mask = (v_channel >= 1) & (v_channel <= 254)

    edges = cv2.Canny(v_channel.astype(np.uint8), threshold1 = 20, threshold2 = 80)

    valid_edges = (edges > 0) & valid_mask

    score = valid_edges.sum() / valid_mask.sum()

    return float(score)

def analyzeImage(imagePath: str):
    img_bgr = cv2.imread(imagePath) # OpenCV loads as BGR, not RGB
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    v_flat = img_hsv[:, :, 2].flatten()
    # flatten all pixels as flat arrays immediately, and then check for value (brightness) issues
    # if the value of the pixel is under 20 or over 240 its unreliable.
    valid_mask = (v_flat >= 20) & (v_flat <= 240)
    h_flat = img_hsv[:, :, 0].flatten()[valid_mask]
    s_flat = img_hsv[:, :, 1].flatten()[valid_mask]
    v_flat = img_hsv[:, :, 2].flatten()[valid_mask]

    # matching the style of HUE_Buckets and SAT_Buckets, use those flat arrays to input the number of pixels that fit into each one
    # so for example, returnArray[0][0] would be hue from 0-14 and saturation from 0-24.
    # returnArray[2][3] woudl be hue from 45-59, and saturation from 50-74.
    # each row is saturation. Each column is heu
    returnArray = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # Determine which hue/sats bucket each pixel belongs to.
    h_indices = np.digitize(h_flat, HUE_Buckets) - 1
    s_indices = np.digitize(s_flat, SAT_Buckets) - 1

    # clamp indices to valid range so bad index errors dont occur
    h_indices = np.clip(h_indices, 0, len(HUE_Buckets) - 1)
    s_indices = np.clip(s_indices, 0, len(SAT_Buckets) - 1)

    totalAnalyzedPixels = 0;

    # Loop over every pixel and increment it's corresponding cell in returnArray.
    for s_idx, h_idx in zip(s_indices, h_indices):
        returnArray[s_idx][h_idx] += 1
        totalAnalyzedPixels += 1
    
    # round and set tiny amounts to 0 for simplicity
    for i, row in enumerate(returnArray):
        for j, col in enumerate(row):
            returnArray[i][j] /= totalAnalyzedPixels
            returnArray[i][j] = round(returnArray[i][j], 4)
            if (returnArray[i][j] < 0.0001):
                returnArray[i][j] = 0.0
    
    return returnArray

# now using earth movers distance
# Adapted to be usable for 1D arrays as well
def compareImageDistrs(array1, array2):
    # Convert inputs to numpy arrays to easily check dimensions
    arr1 = np.array(array1)
    arr2 = np.array(array2)
    
    # Utilize absolute distances as a fall back
    if arr1.size == 0 or arr2.size == 0:
        print("Signatures empty. use absolute dist. for this one")
        return float(np.sum(np.abs(arr1.flatten() - arr2.flatten())))

    sig1 = []
    sig2 = []

    # if the Input array is 1D, that must mean that it's from the texture analysis thing.
    if arr1.ndim == 1:
        for i, val in enumerate(arr1):
            if val > 0.0:
                sig1.append([float(val), float(i)])
                
        for i, val in enumerate(arr2):
            if val > 0.0:
                sig2.append([float(val), float(i)])

    # If the input is 2D, then that means that is from analyze image with saturation and hue
    else:
        for i, row in enumerate(arr1):
            for j, val in enumerate(row):
                if val > 0.0:
                    sig1.append([float(val), float(i), float(j)]) # Weights now as [Y, X]
                    
        for i, row in enumerate(arr2):
            for j, val in enumerate(row):
                if val > 0.0:
                    sig2.append([float(val), float(i), float(j)])

    sig1 = np.array(sig1, dtype=np.float32)
    sig2 = np.array(sig2, dtype=np.float32)

    # Pick out earth mover distance from OpenCV's response
    emd_result = cv2.EMD(sig1, sig2, cv2.DIST_L2)
    return emd_result[0]

def visualizeImageDistrDifferences(array1, array2):
    returnArray = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i, row in enumerate(array1):
        for j, col in enumerate(row):
            returnArray[i][j] = abs(array1[i][j] - array2[i][j])

    return returnArray

# Classification plan.
# step 1, go through each reference image and cache their results once I move to google colab
# because we're gonna be going through a ton of images and averaging them out
canopyArray = analyzeImage(CANOPY_REF)
lowShrubArray = analyzeImage(LOWSHRUB_REF)
patchyArray = analyzeImage(PATCHY_REF)
dryArray = analyzeImage(DRY_REF)

inputArray = analyzeImage(IMAGE_PATH)

canopyNoise = valueNoise(CANOPY_REF)
lowShrubNoise = valueNoise(LOWSHRUB_REF)
patchyNoise = valueNoise(PATCHY_REF)
dryNoise = valueNoise(DRY_REF)
inputNoise = valueNoise(IMAGE_PATH)

print(f"Input noise level: {inputNoise}")
print(f"Canopy noise level: {canopyNoise}")
print(f"Low shrub noise level: {lowShrubNoise}")

refNoises = [canopyNoise, lowShrubNoise, patchyNoise, dryNoise]
labels = ["canopy", "lowShrub", "patchy", "dry"]

colorDistances = [compareImageDistrs(inputArray, canopyArray), compareImageDistrs(inputArray, lowShrubArray),
                  compareImageDistrs(inputArray, patchyArray), compareImageDistrs(inputArray, dryArray)]

noiseDistances = [abs(inputNoise - r) for r in refNoises]

closestTwo = heapq.nsmallest(2, colorDistances)
closest, second_closest = closestTwo
confidence = (second_closest - closest) / second_closest

# Noise is weighted heavier by default (BASE_NOISE_WEIGHT).
# When color confidence is low, noise weight increases further toward 1.0.
BASE_NOISE_WEIGHT = 0.65
colorWeight = (1 - BASE_NOISE_WEIGHT) * confidence
noiseWeight = 1 - colorWeight

combined = [colorWeight * colorDistances[i] + noiseWeight * noiseDistances[i] for i in range(4)]

bestIdx = combined.index(min(combined))
closestTwo = heapq.nsmallest(2, combined)
finalConfidence = (closestTwo[1] - closestTwo[0]) / closestTwo[1]

print(f"{labels[bestIdx]} ({finalConfidence:.0%} confident)")





# for visualization
def getResults():
    inputArray = analyzeImage(IMAGE_PATH)
    canopyArray = analyzeImage(CANOPY_REF)
    lowShrubArray = analyzeImage(LOWSHRUB_REF)
    patchyArray = analyzeImage(PATCHY_REF)
    dryArray = analyzeImage(DRY_REF)
    return inputArray, canopyArray, lowShrubArray, patchyArray, dryArray

if __name__ == "__main__":
    inputArray, canopyArray, lowShrubArray, patchyArray, dryArray = getResults()
    for row in inputArray:
        print(row)
        