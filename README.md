# A beginner's guide to machine learning

## Introduction
The Internet is full of great open-source materials, books, blogs and articles in this topic, and it is not easy to decide what to read first if you are a beginner. I have decided to learn about AI and ML a few months ago, now my aim is to help people who are also novices but interested in these topics. I will collect the links of everything I read with short desciptions. I intend to structure the content by topics from the basics to the more advanced readings.

I'm planning to continuosly update this list of resources as I find more interesting readings.

## Programing languages, frameworks, libraries
While you are learning the basics I think it is not useful to dig deep into any of the deep learning frameworks, they hide the subtle details of all the intersting stuff which I want to understand to the smallest details before I go on with the advanced use cases. It is better to start with a library that provides only the necessary linear algebra.

Using python for experimenting with gradient descent and back-propagation is very convenient. Numpy performs vector and matrix operations very efficiently, because under the hood it uses one of C libraries to do the math. In python you can very conveniently visualize the data you are working on with one of the plotting libraries like matplotlib or pyplot but there are many other choices, to get a good overview of them I recommend to watch the talk from PyCon2017.
  
  - Linear algebra libraries:
    - C/C++: [OpenBLAS](http://www.openblas.net/), [boost uBLAS](http://www.boost.org/doc/libs/1_65_1/libs/numeric/ublas/doc/index.html), [Intel Math Kernel Library](https://software.intel.com/en-us/mkl), [Eigen](https://eigen.tuxfamily.org/dox/) 
    - python: [numpy](http://www.numpy.org/), [scipy](https://scipy.org/), [scikit-learn](http://scikit-learn.org/stable/)
  - Data visualization, charts, graphs, 3D wireframes and surface rendering:
    - [The Python Visualization Landscape PyCon 2017 *by Jake VanderPlas*](https://youtu.be/FytuB8nFHPQ)
    - [matplotlib](https://matplotlib.org/) comes with scipy
  - Most popular deep learning frameworks:
    - Caffe2
    - TensorFlow
    - PyTorch
    - Theano

## Math background

  The Deep learning book has a good summary on [Linear Algebra](http://www.deeplearningbook.org/contents/linear_algebra.html) and [Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html) which you will find very useful if you have learnt these a long time ago.
  
  On Colah's blog you can find [Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/), a very well illustrated post about probability distributions and entropy. 

## Optimization
- Linear regression
  - [Deep learning book 5.1.4 (page 105)](http://www.deeplearningbook.org/contents/ml.html)
- Linear least squares
  - [On Wikipedia](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Weighted_linear_least_squares)
- Multivariate linear regression using gradient descent
  - [python code](https://gist.github.com/samueljackson92/8148506)
- Polynomial fitting
  - numpy.polyfit
- Multivariate polynomial fitting
  - [python code](https://github.com/mrocklin/multipolyfit)

## Neural network basics
- Types of artificial neurons:
  - [Perceptrons](http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons)
  - [Sigmoid neurons](http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons)
  - [Rectified linear units (page 170)](http://www.deeplearningbook.org/contents/mlp.html)
- Gradient descent
  - From [Chapter 1 of Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent) you can familiarize yourself with gradient descent. You will get to know what makes its stochastic version stochastic, what is a mini-batch, what are hyperparameters.
   - [Numerical computations](http://www.deeplearningbook.org/contents/numerical.html)
- Back propagation
  - For an introductory reading I also recommend [Chapter 2 of Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/chap2.html)
  - You also have to read [Calculus on Computational Graphs: Backpropagation *on Colah's blog*](http://colah.github.io/posts/2015-08-Backprop/), it has beautiful illustrations that really helps imagine the process.

## ???


## Common abbreviations
- AI - Artifical Inteligence
- ML - Machine Learning
- ANN - Artificial Neural Network
- DNN - Deep Neural Network: NN with many hidden layers)
- CNN - Convolutional Neural Network
- RNN - Recurrent Neural Network
- MLP - Multi Layer Perceptron
- ReLU - Rectified Linear Unit
- SGD - Stochastic Gradient Descent

## References - Books
- Neural Networks and Deep Learning *by Michael Nielsen* [[read online]](http://neuralnetworksanddeeplearning.com) [[Examples in python]](https://github.com/mnielsen/neural-networks-and-deep-learning) [[Example in C++/uBLAS]](https://github.com/GarethRichards/Machine-Learning-CPP/blob/master/README.md)
- Deep Learning *by Ian Goodfellow and Yoshua Bengio and Aaron Courville* [[read online]](http://www.deeplearningbook.org) [[amazon]](https://www.amazon.com/dp/0262035618/)
- The Data Science Design Manual *by Steven S. Skiena* [[amazon]](https://www.amazon.com/dp/3319554433/)

## References - Blogs
- [Colah's blog](http://colah.github.io/)
- [Andrej Karpathy's blog](http://karpathy.github.io/)

## Other Links
- [List of datasets for machine learning](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)
- [KD nuggets](https://www.kdnuggets.com/)
- [Kaggle](https://www.kaggle.com/)

## Online courses
- [A collection of 17 popular courses](https://www.marketingaiinstitute.com/blog/17-artificial-intelligence-courses-to-take-online)
- [Coursera ML foundations](https://www.coursera.org/learn/ml-foundations)
- [Coursera Neural networks and deep learning](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Udacity Deep learning](https://www.udacity.com/course/deep-learning--ud730)
- [Udacity ML nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009#)
