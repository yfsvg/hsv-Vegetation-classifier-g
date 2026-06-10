# Research into relevant topics
List of relevant papers and research that are gonna help me figure out where to start and which directions to take.

## Promising papers -
* https://www.mdpi.com/2313-433X/12/1/9  Accurate Segmentation of Vegetation in UAV Desert Imagery Using HSV-GLCM Features and SVM Classification 
* https://jurnal.umsu.ac.id:444/index.php/jcositte/article/view/26495 
https://www.mdpi.com/1424-8220/18/4/1253  A New Vegetation Segmentation Approach for Cropped Fields Based on Threshold Detection from Hue Histogram (Very similar)
Differences is that they are doing segmentation rather than classification
They use SVM on labeled data, i use very few curated reference images instead
They don't actually use a histogram in their PROCESS that they made
* https://link.springer.com/chapter/10.1007/978-981-16-8225-4_21 HsvGvas: HSV Color Model to Recognize Greenness of Forest Land for the Estimation of Change in the Vegetation Areas
* https://bioone.org/journals/journal-of-resources-and-ecology/volume-16/issue-4/j.issn.1674-764x.2025.04.002/Identification-of-Grassland-Vegetation-Coverage-and-Height-Based-on-Vegetation/10.5814/j.issn.1674-764x.2025.04.002.full (Just in the general direction of HSV)
* https://www.sciencedirect.com/science/article/pii/S2214317315000347 Greenness identification based on HSV decision tree
* https://diglib.eg.org/server/api/core/bitstreams/f24394e8-25f0-42fe-a21c-58d8bf5107f5/content 

# Summaries
**Automatic Identification of Grassland Vegetation Coverage and Height Using Visible Light Vegetation Indices**
- This one proposes methods for automatic grassland vegetation monitoring using drone monitoring and mobile smartphone systems. They use vegetation coverage extraction with five visible light vegetation indices (EXG, EXGR, NGRDI, GLI, RGBVI) and Otsu’s method.They then found grass layer height measurements using HSV color space conversion to identify red calibration rings on a custom measuring pole. 
- Found that this method could achieve around ~90% accuracy. This is an example of how pure HSV algorithms can be used effectively in order to make conclusions about vegetation, and there is no need for massive datasets. Also shows that using HSV is better than RGB since you dont need to use illumination normalization

**Accurate Segmentation of Vegetation in UAV Desert Imagery Using HSV-GLCM Features and SVM Classification**
https://www.mdpi.com/2313-433X/12/1/9
- This study shows a machine learning approach to segment green vegetation from high-resolution UAV drone imagery (Similar to the YOLOv8 im comparing to). It extracts color features by converting images to the HSV color space and gets textural details using a "Gray-Level Co-occurrence Matrix". These features are then fed into a svm classifier to segment the vegetation.
- This directly connects to this study because it demonstrates how HSV color features + texture analysis (The two pillars of my system) can be used to distinguish vegetation. Also addresses the trade off in the README regarding the tradeoffs compared to ML algos (They need big databases, black box, etc.). This already gets 91% accuracy while remaining practical ad nsmall. However AGAIn note that this is for segmentation rather than classification, a big difference

**Herbal Plant Image Retrieval Using HSV Color Histogram and Random Forest Algorithm**
- This paper designs an image search and classification system for herbal plant leaves based on color similarity. It extracts color features by building a 512-dimensional HSV color histogram (8^3 bin configuration, similar to what I'm doing) and uses a Random Forest classifier to identify the plant species.
- Almost perfectly matches this algorithm without the texture step. Their system achieved a 95.56% accuracy rate, demonstrating that similar systems can work. The authors highlight that a big advantage of this methodology is the simplicity of the model yet the high efficiency, which is what i highlghed at the start.

**HsvGvas: HSV Color Model to Recognize Greenness of Forest Land for the Estimation of Change in the Vegetation Areas**
- This paper uses the HSV color space to evaluate the "greenness" percentage of a given geographic region and track changes over time. They turn it into a program that they called HsvGvas.
- Again, this shows an example of how we can use the specific benefits of HSV color space for assessing plant coverage and vegetation health, which is incredibly similar to what im doing other than the histogram part. The seperation of hue and luminance makes it incredibly easy again.

**A New Vegetation Segmentation Approach for Cropped Fields Based on Threshold Detection from Hue Histograms**
- This study establishes a vegetation segmentation method for low-cost UAV imagery by analyzing HSV hue histograms, which is what I am planning on using too. Based on the distribution of colors in agricultural fields, it uses the histogram to dynamically detect threshold values that can help segment out vegetation.
- This again validates the use of localized color histograms to make decisions about vegetation classification. Also it is explicitly stated the key reasons as to why histogram based thresholding was used (Interpretability, computational efficiency). Again, note that this is SEGMENTATION rather than classification. Final 87.29% mean accuracy, which is decent.

**Color Style Transfer Techniques using Hue, Lightness and Saturation Histogram Matching**
- This study develops an image transformation technique that matches 3D color distributions by analyzing histograms sequentially across HSV cannels. It focuses on transferring artistic styles and color atmospheres between images without needing structural segmentation.
- Not as hugely connected. While their objective is largely style transfer and mine is classification, both of these methodlogyes rely on treating color distributions as statistical profiles. They found that ingoring spatial coherence can introduce noise, proving why this model needs texture measurements as a secondary anchor

**Greenness identification based on HSV decision tree**
- This paper made a threshold based decision tree that uses the HSV color space to isolate crops from backgrounds and changing weather conditions. Also directly uses color rather than infrared light, which is one of my main benefits
- This directly aligns with my study's focus on making an algorithm that does a lot for interpretability, efficiency, and needing very little data. The authors literally criticize traditional spectral indices for failing under uneven lighting, and then state that an HSV decision tree would provide a better rule based path, and also be interpretable for why it came to a decision. Very little computations, no training overhead too. Claimed 100% accuracy.