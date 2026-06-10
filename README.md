The purpose of this paper is to see whether or not a custom algorithm that relies purely on color images can identify vegetation health, how accurately it can do so, how resource efficient such an algorithm is, and how it compares to a YOLOv8 model specifically fine-tuned to identify and categorize these images.

The way that the algorithm works is by essentially creating a color histogram of each pixel in the image, and comparing it to a list of reference images that the user categorizes themselves. The distribution that is the closest are the one that gets categorized, and the more distinctly similar the input distribution is to one of the reference distributions, the more “confident” the model is. If the user provides multiple reference images, then the reference for each category is averaged.

In addition, the texture of each image is also measured in terms of how significant shifts in value are in HSV, symbolizing large spots of shadows and highlights.

There are two immediate improvements this has over a computer vision model such as YOLOv8. Firstly, you can provide as few reference images as 1 for each category for this algorithm to work, and have the choice to add more to be more accurate. On the other hand, a model such as YOLOv8 often requires hundreds of images for each category to work properly. 

In addition, this algorithm is clear and interpretable, and you can analyze each step of the way it takes to get to its final conclusion. On the other hand, a model such as YOLOv8 is essentially a black box for most users.
