# keras-wedge
![wedge image](pics/wedge_4_100.png)

Design, implementation and documentation of Wedge Dropout.

## Install

```bash
pip install keras-wedge
```

## Documentation

Wedge Dropout implements a recent technique in Convolutional Network Design: critiquing feature maps. Wedge Dropout analyzes all of the feature maps created by a CNN and contributes negative feedback to those feature maps that fail a test. This has the effect of improving the CNN's performance because the analysis checks a basic quality of feature maps: decorrelation.

See this notebook for a basic explanation of the concept:
![Wedge Dropout Intro](https://github.com/LanceNorskog/keras-wedge/blob/main/Wedge%20Dropout%20Introduction.ipynb%20-%20Colaboratory.pdf)

See this notebook for a demonstration of our analysis function in a simple CNN:
![similarity notebook pdf](https://github.com/LanceNorskog/keras-wedge/blob/main/Similarity%20mnist_convnet%20-%20Colaboratory.pdf)

