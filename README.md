# Digit recognition problem using the MNIST
 Digit recognition with MNIST using linear and logistic regression, non-linear feature, regularization and kernel 

# Objective
The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, we will get a chance to experiment with the task of classifying these images into the correct digit using several methods.

# Structure of project
- ```linear_regression.py```: where we will implement linear regression
- ```svm.py``` where we will implement support vector machine
- ```softmax.py`` where we will implement multinomial regression
- ```features.py``` where we will implement principal component analysis (PCA) dimensionality reduction
- ```kernel.py``` where we will implement polynomial and Gaussian RBF kernels
- ```main.py``` where we will use the code you write for this part of the project

## Dimensionality Reduction via PCA

Principal Components Analysis (PCA) is the most popular method for linear dimension reduction of data and is widely used in data analysis. For an in-depth exposition see [here](https://online.stat.psu.edu/stat505/lesson/11.) 

Briefly, this method finds (orthogonal) directions of maximal variation in the data. By projecting an n x d dataset X onto $k \leq d$  of these directions, we get a new dataset of lower dimension that reflects more variation in the original data than any other k-dimensional linear projection of X. By going through some linear algebra, it can be proven that these directions are equal to the k eigenvectors corresponding to the k largest eigenvalues of the covariance matrix $\widetilde{X}^T\widetilde{X}$, where $\widetilde{X}^T$  is a centered version of our original data.

## Cubic Features

In this section, we will also work with a cubic feature mapping which maps an input vector $x=[x_1,x_2,...,x_d]$  into a new feature vector $\phi(x)$, defined so that for any $x,x' \in \mathbb{R}^d$:

$$
\phi (x)^T \phi (x) = (x^{T}x'+1)^3
$$

