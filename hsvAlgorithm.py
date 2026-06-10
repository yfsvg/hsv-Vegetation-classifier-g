from PIL import Image
import cv2
import numpy as np

# Image that is to be analyzed
IMAGE_PATH = "textureTestBad.png"

# Ref images
GOOD_REF_PATH = "referenceImages/goodRef.png"
MID_REF_PATH = "referenceImages/mediocreRef.png"
POOR_REF_PATH = "referenceImages/poorRef.png"

# 12 buckets, each containing a different grouping of colors, with steps of 14.
# these are all maximum HSV values, so for example "0" represents 0-14 range. "120" is 120-134.
HUE_Buckets = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168];

# 4 categorizes for saturation. each step is 24 this time
SAT_Buckets = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5]

# takes in the 2D value array
def gradientStrengthDetection(valueArray):
    # RESIZE! I figured this out to be the issue. The images were at different resolutions.
    resized_array = cv2.resize(valueArray, (500, 500), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized_array, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    return laplacian.var()


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

    gradientStrength = gradientStrengthDetection(img_hsv[:, :, 2])
    print(f"[{imagePath}] Gradient Strength: {gradientStrength:.4f}")

    # matching the style of HUE_Buckets and SAT_Buckets, use those flat arrays to input the number of pixels that fit into each bucket
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

# Since areas with lots of vegetation are more likely to have things like trees and shrubbery
# that create large areas of shade, we look at how "uneven" the ground is by using laplacian variance
# In wild forests (What this algo is for), flat areas actually really aren't that great, so higher
# variance means more healthy. 
# NOTE: no longer in use
def homemadeTextureAnalysis(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # valueDistributionBuckets are split into 8 like [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    hist = cv2.calcHist([blurred], [0], None, [8], [0, 256])

    # Normalize histogram to ensure that sum of all bins will now equal 1.0 regardless of resolution
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist /= total_pixels

    # flatten and return
    valueHistogram = hist.flatten().tolist()
    return valueHistogram

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


inputArray = analyzeImage(IMAGE_PATH)

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

def getResults():
    inputArray = analyzeImage(IMAGE_PATH)
    goodArray = analyzeImage(GOOD_REF_PATH)
    mediocreArray = analyzeImage(MID_REF_PATH)
    poorArray = analyzeImage(POOR_REF_PATH)
    return inputArray, goodArray, mediocreArray, poorArray

if __name__ == "__main__":
    inputArray, goodArray, mediocreArray, poorArray = getResults()
    for row in inputArray:
        print(row)


# New plan for a proper texture analysis.
# Keep the value as a 2D array. Blur it a TINY bit to get rid of pixel variations and only have spatial patterns.
# run an edge detection thing to find gradient shifts, bigger gradient shifts meaning
# Use more commonly when there is less confidence with the hue.