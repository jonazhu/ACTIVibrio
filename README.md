# ACTIVibrio
ACTIVibrio is the combination of active learning techniques with Vibrio cholerae video embedding data, where we specifically investigate stopping criteria for a variety of active learning algorithms, in a project by Darin Boyes, Shivank Sadasivan, and Jonathan Zhu for 02-750 Automation of Scientific Research at CMU.


# Learned Representation from Videos Using Vision Transformer
## Directory structure
Overview of directory structure can be found in tree.txt file.
### Parent directory of processed videos 
Date_Time_Magnification_Discontinuous_Drawer Day-Month-Year Hours-Minutes-Seconds

Example:
```
241010_105227_4x_10x_20x_40x_Discontinuous_Drawer1 10-Oct-2024 10-48-54
```
Each processed video is found in results_images subdirectory.

Each of these directories contains videos (25 frames each) of biofilm development of different mutants of Vibrio cholerae. Processed videos are in the results_images subdirectory, and are processed using subpixel registration and smoothed using a local Gaussian filter.

### Output embeddings
Embeddings directory contains generated 1x512-dimension embeddings of 10x magnification videos for each mutant used in analysis (144 embbeddings per mutant). Embeddings are obtained by first augmenting the output of the Vision Transformer to a linear projection head with 512 dimension output, passing individual frames through the model to get (25x512) embeddings, then using a fixed positional encoding scheme to condense the 25x512 embeddings to a 1x512 embedding. The fixed positional encoding scheme (using dimension=D) is given by:

$PE(pos,2i) = sin\left(\frac{pos}{10000^\frac{2i}{D}}\right)$

$PE(pos,2i + 1) = cos\left(\frac{pos}{10000^\frac{2i}{D}}\right)$

