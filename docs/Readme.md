# Introduction
Convolutional Neural Networks (CNNs) operate by creating a set of grayscale images (called *feature maps*) that correspond to features in an input image. Given a picture of a cat, one feature map may describe the cat's nose, another may describe the cat's left ear, and a third may describe the horizontal stripes across the cat's face.

Here is a heatmap showing correlation across all 64 feature maps for a very simple Convolutional Neural Network:

![heatmap](fmap%20similarity%20epochs%20vs%20fmaps.png)

At the start of training, the feature maps are strongly correlated (magenta) and become less correlated (blue) as training continues.
# Coincidence v.s. Causality
There various options for the causality involved between decorrelation and performance (predictive power):
1. No causality- decorrelation during training is not related to the improvement in performance, they are merely coincidental.
2. Improvement in performance causes decorrelation
3. Decorrelation causes improvement in performance
4. Both #2 and #3
5. Performance and Decorrelation are both caused by external factor(s)

I do not see how to create an experiment that proves #1, #2, #4 or #5. This project creates an experiment to test #3. We will test #3 by altering a convolutional 
neural network to add a measurement of correlation to the loss function. 

# Possible Designs of Experiments
There are a few ways to alter a CNN so that a measurement of correlation adds signal to the loss function.
1. DIRECT: Measure correlation across feature maps and add measured value directly to the loss function
2. INDIRECT: Measure correlation across feature maps and interfere with feature maps that violate a criterion

NEED DIAGRAMS OF EACH APPROACH

Both methods seem like they should work. But, the two methods have a subtle difference in the propagation of causality. 
There are many feature maps, but only some are correlated. 
The direct method does not inform the training feedback loop as to which feature maps are correlated. 
The indirect method does give this information to the training feedback loop.
It is possible that this difference makes the indirect method more effective than the direct method.
I have not tested the direct method, since the results from Wedge Dropout are encouraging.

# Design of Wedge Dropout is Indirect

Wedge Dropout is an implementation of the indirect method. It works by 
1. randomly selecting pairs of the feature maps created for each training sample, 
2. making a binary decision about whether pairs are strongly correlated, and
3. zeroing out those feature maps which fail the test. 

This design was inspired by the SpatialDropout technique, which randomly zeroes out some feature maps.
