The purpose of this paper is to see whether or not a custom algorithm that relies purely on color images can identify vegetation health, how accurately it can do so, how resource efficient such an algorithm is, and how it compares to a YOLOv8 model specifically fine-tuned to identify and categorize these images.

The way that the algorithm works is by essentially creating a color histogram of each pixel in the image, and comparing it to a list of reference images that the user categorizes themselves. The distribution that is the closest are the one that gets categorized, and the more distinctly similar the input distribution is to one of the reference distributions, the more “confident” the model is. If the user provides multiple reference images, then the reference for each category is averaged.

In addition, the texture of each image is also measured in terms of how significant shifts in value are in HSV, symbolizing large spots of shadows and highlights.

There are two immediate improvements this has over a computer vision model such as YOLOv8. Firstly, you can provide as few reference images as 1 for each category for this algorithm to work, and have the choice to add more to be more accurate. On the other hand, a model such as YOLOv8 often requires hundreds of images for each category to work properly. 

In addition, this algorithm is clear and interpretable, and you can analyze each step of the way it takes to get to its final conclusion. On the other hand, a model such as YOLOv8 is essentially a black box for most users.

The purpose of the value check is that in forestry applications, high spectral greenness concentrations can create false-positive healthy classifications when encountering low-lying grasslands or plains, which is actually not good since those dont have lots of shrubbery and trees. To go against this this domain-specific issue, this pipeline implements a concentration-dependent Structural weighting layer with the value check. The weight of the spatial Value texture metric is scaled as a continuous function of total green pixel density. This forces the system to demand progressively stricter high-frequency shadow signatures as green density increases, which in theory could isolate flat plains from complicated tree lines and canopy covers.




Categories and their definitions
Canopy - Tree cover
Dry - Dry/bare, very litte vegetation
Flat - Green vegetation, no tree cover
Patchy - Any kind of mixed, not just strictly patchy

The value filter should check how abrupt the disruptions are if its primarily green to identify if its flat or canopy

If yellow and green are the most dominant...
Distinguishing between dry and patchy can rely on how clustered together the mixes of yellow and green are.
If its more clustered then its more likely to be patchy


Notes:
1. Go back through all of them I marked as dry and see if I marked some that are patchy as poor
2. Search specifically for flat images




RULES for categories
Canopy - Fully developed, almost entirely trees
Dry - Very little vegetation, bare, or entirely yellow
Patchy - Mix of dry/bare and healthy vegetation
LowShrub - Health vegetation that ISNT fully developed treetop

Lowshrub and canopy all have UNDENIABLY majority green covers
Patchy is if there is still some vegetation
Dry is dominated by dirt/brown MAJORITY