# A beginner's guide to machine learning

## Introduction
The Web is full of great books, blogs, articles and open-source codes in this topic, and it is not easy to decide what to read first if you are a beginner. I have decided to learn about AI and ML a few months ago, now my aim is to help people who are also novices but interested in these topics. I will collect the links of everything I read with short desciptions. I will structure the content by topics from the basics to the more advanced readings.

I'm planning to continuously update this list of resources when I find more notable readings.

## Programming languages, frameworks, libraries
While you are learning the basics I think it is not useful to dig deep into any of the deep learning frameworks. Those operate on higher levels while they hide the subtle details of all the intersting stuff. My goal is to understand the fundamentals of neural networks to the smallest details before I go on with the advanced use cases. With this attitude it was better to start with a library that provides only the necessary linear algebra.

Using python for experimenting with gradient descent and back-propagation is very convenient. Numpy performs vector and matrix operations very efficiently, because under the hood it uses one of C libraries to do the math. In python you can easily visualize the data you are working on with one of the plotting libraries like matplotlib or pyplot. There are many other choices for drawing different charts, to get a good overview of them I recommend to watch the talk from PyCon2017 (referenced below).
  
  - Linear algebra libraries:
    - C/C++: [OpenBLAS](http://www.openblas.net/), [boost uBLAS](http://www.boost.org/doc/libs/1_65_1/libs/numeric/ublas/doc/index.html), [Intel Math Kernel Library](https://software.intel.com/en-us/mkl), [Eigen](https://eigen.tuxfamily.org/dox/) 
    - python: [numpy](http://www.numpy.org/), [scipy](https://scipy.org/), [scikit-learn](http://scikit-learn.org/stable/)
    - GPU accelerated: [NVIDIA Deep Learning SDK](https://developer.nvidia.com/deep-learning-software)
  - Data visualization, charts, graphs, 3D wireframes and surface rendering:
    - [The Python Visualization Landscape PyCon 2017 *by Jake VanderPlas*](https://youtu.be/FytuB8nFHPQ)
    - [matplotlib](https://matplotlib.org/) comes with scipy
  - Most popular deep learning frameworks:
    - [Caffe2](https://www.caffe2.ai/)
    - [TensorFlow](https://www.tensorflow.org/)
    - [PyTorch](http://pytorch.org/)
    - [Theano](http://deeplearning.net/software/theano/)

## Where to start

The first book I started to read was Ian Goodfellow's [DeepLearningBook.org] which is scientifically very thorough, very dense and long book, thus it is not an easy reading.
The first part (chapter 2-5) of the book covers the required theoretical background, and continues with the basic concepts of learing algorithms, which I found very useful to refresh my knowledge. The second part starts with [Deep feedforward networks](http://www.deeplearningbook.org/contents/mlp.html) and continues with [Regularization for Deep Learning](http://www.deeplearningbook.org/contents/regularization.html), during this chapter I started to feel a little bit lost in theory.

So I've looked for some other reading with more practical examples, which could show me how is everything applied that I have already read. Then I found the book of Michael Nielsen - [NeuralNetworksAndDeepLearning.com], which was exacly what I was looking for.
It is going through the theory step by step while implementing a Python/NumPy based solution with examples, charts, explanations to practical questions. At this point I decided to suspend the reading of the Deep Learning book to gain some hands-on experience with the latter. But, of course I intend to continue with Ian Goodfellow's book also, when I will already feel the confidence with the basics.

## Math background

  The Deep learning book has a good summary on [Linear Algebra](http://www.deeplearningbook.org/contents/linear_algebra.html) and [Probability and Information Theory](http://www.deeplearningbook.org/contents/prob.html) which you will find very useful if you have learnt these a long time ago.
  
  On Colah's blog you can find [Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/), a very well illustrated post about probability distributions and entropy. 

## Optimization
- Linear regression
  - [Deep learning book 5.1.4 (page 105)](http://www.deeplearningbook.org/contents/ml.html)
- Least squares polynomial fitting
  - [On Wikipedia](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Weighted_linear_least_squares)
  - [numpy.polyfit](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polyfit.html) and [numpy.polyval](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.polyval.html)
  - [Multivariate polynomial fitting](https://github.com/mrocklin/multipolyfit)
- Multivariate linear regression using gradient descent
  - [python code](https://gist.github.com/samueljackson92/8148506)

## Neural networks
- Types of artificial neurons:
  - [Perceptrons](http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons)
  - [Sigmoid neurons](http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons)
  - [Rectified linear units (page 170)](http://www.deeplearningbook.org/contents/mlp.html)
  - [Hyperbolic tangent and ReLU](http://neuralnetworksanddeeplearning.com/chap3.html#other_models_of_artificial_neuron)
- Gradient descent
  - Reading [Chapter 1 of Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent) you can familiarize yourself with gradient descent. You will get to know what makes its stochastic version stochastic, what a mini-batch is or what the hyperparameters are.
   - [Numerical computations](http://www.deeplearningbook.org/contents/numerical.html)
- Back propagation
  - For an introductory reading I recommend [Chapter 2 of Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/chap2.html)
  - You also have to read [Calculus on Computational Graphs: Backpropagation *on Colah's blog*](http://colah.github.io/posts/2015-08-Backprop/), it has beautiful illustrations that really helps imagine the process.
- [How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters)

## Other types of machine learning
- Overviews
  - [Intel AI Academy - Machine Learning 101](https://software.intel.com/ai-academy/students/kits/machine-learning-101)
  - [scikit-learn - Supervised learning](http://scikit-learn.org/stable/supervised_learning.html)
- Naive Bayes
  - [SciKit Learn](http://scikit-learn.org/stable/modules/naive_bayes.html)
  - [Mathematical Concepts and Principles of Naive Bayes](https://software.intel.com/articles/mathematical-concepts-and-principles-of-naive-bayes)
- Support vector machines
  - [On Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
  - [SciKit Learn](http://scikit-learn.org/stable/modules/svm.html)
  - [Understanding Support Vector Machine algorithm from examples (along with code)](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)

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
- SVM - Support Vector Machine

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
- [DeepLearning.net](http://deeplearning.net/)
- [Deep learning library](https://www.cse.iitk.ac.in/users/sigml/lec/DeepLearningLib.pdf)

## Online courses
- [A collection of 17 popular courses](https://www.marketingaiinstitute.com/blog/17-artificial-intelligence-courses-to-take-online)
- [Coursera ML foundations](https://www.coursera.org/learn/ml-foundations)
- [Coursera Neural networks and deep learning](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Udacity Deep learning](https://www.udacity.com/course/deep-learning--ud730)
- [Udacity ML nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009#)
