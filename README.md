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

# Stopping Criteria

This project focuses also on stopping criteria for various active learning query selection algorithms: these include the high-performance heuristics of uncertainty sampling and query by committee (QBC), the Type I algorithm with theoretical guarantees Importance-Weighted Active Learning (IWAL), the Type II algorithms with theoretical guarantees DH and PLAL, and an exploration of deep learning selection. 

## Uncertainty Sampling

This method simply selects the point whose classification is most uncertain by the current model. Stopping criteria have already been proposed, and these are confidence-based stopping (stop when average uncertainty falls below a specific threshold) and gradient-based stopping (stop when difference between median uncertainty between iterations falls below a specific threshold).

## Query by Committee

QBC aims to establish a committee of models; these models predict on unlabeled data, and we select the point on which the committee has the highest disagreement. One stopping criteria has been proposed, which stops when average disagreement falls below a specific threshold or approaches a gradient of 0. We propose distribution-based stopping, which stops when the distributions of the data used for each committee member are highly similar. 

## Importance-Weighted Active Learning

(info to be added)

## DH Algorithm

(info to be added)

## PLAL Algorithm

(info to be added)

## Deep Learning

(info to be added)

