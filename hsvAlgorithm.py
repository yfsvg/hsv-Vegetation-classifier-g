from PIL import Image
import cv2
import numpy as np

# Image that is to be analyzed
IMAGE_PATH = "bare.png"

# Ref images
GOOD_REF_PATH = "referenceImages/goodRef.png"
MID_REF_PATH = "referenceImages/mediocreRef.png"
POOR_REF_PATH = "referenceImages/poorRef.png"

# 12 buckets, each containing a different grouping of colors, with steps of 14.
# these are all maximum HSV values, so for example "0" represents 0-14 range. "120" is 120-134.
HUE_Buckets = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165];

# 4 categorizes for saturation. each step is 24 this time
SAT_Buckets = [0, 25, 50, 75]

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

    # matching the style of HUE_Buckets and SAT_Buckets, use those flat arrays to input the number of pixels that fit into each bucket
    # so for example, returnArray[0][0] would be hue from 0-14 and saturation from 0-24.
    # returnArray[2][3] woudl be hue from 45-59, and saturation from 50-74.
    # each row is saturation. Each column is heu
    returnArray = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    # Determine which hue/sats bucket each pixel belongs to.
    h_indices = np.digitize(h_flat, HUE_Buckets) - 1
    s_indices = np.digitize(s_flat, SAT_Buckets) - 1

    # clamp indices to valid range so bad index errors dont occur (Exactly on the border, such as h=179 -> error, will try to get index 12)
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


def compareImageDistrs(array1, array2):
    returnDistance = 0;

    for i, row in enumerate(array1):
        for j, col in enumerate(row):
            returnDistance += abs(array1[i][j] - array2[i][j])

    return returnDistance

def visualizeImageDistrDifferences(array1, array2):
    returnArray = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    for i, row in enumerate(array1):
        for j, col in enumerate(row):
            returnArray[i][j] = abs(array1[i][j] - array2[i][j])

    return returnArray



inputArray = analyzeImage(IMAGE_PATH)

for row in inputArray:
    print(row)

goodArray = analyzeImage(GOOD_REF_PATH)
mediocreArray = analyzeImage(MID_REF_PATH)
poorArray = analyzeImage(POOR_REF_PATH)

goodDistance = compareImageDistrs(goodArray, inputArray)
mediocreDistance = compareImageDistrs(mediocreArray, inputArray)
poorDistance = compareImageDistrs(poorArray, inputArray)

closest = min(goodDistance, mediocreDistance, poorDistance)
if closest == goodDistance:
    print("Closest to good")
elif closest == mediocreDistance:
    print("Closest to mediocre")
else:
    print("Closest to poor")