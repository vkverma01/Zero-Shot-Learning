### Note: This code is based on the new split proposed by [Y Xian et. al.](https://arxiv.org/pdf/1707.00600.pdf) 

# A Simple Exponential Family Framework for Zero-Shot Learning

## Abstract:
Abstract. We present a simple generative framework for learning to predict previously
unseen classes, based on estimating class-attribute-gated class-conditional
distributions. We model each class-conditional distribution as an exponential family
distribution and the parameters of the distribution of each seen/unseen class
are defined as functions of the respective observed class attributes. These functions
can be learned using only the seen class data and can be used to predict
the parameters of the class-conditional distribution of each unseen class. Unlike
most existing methods for zero-shot learning that represent classes as fixed embeddings
in some vector space, our generative model naturally represents each
class as a probability distribution. It is simple to implement and also allows leveraging
additional unlabeled data from unseen classes to improve the estimates of
their class-conditional distributions using transductive/semi-supervised learning.
Moreover, it extends seamlessly to few-shot learning by easily updating these
distributions when provided with a small number of additional labelled examples
from unseen classes. Through a comprehensive set of experiments on several
benchmark data sets, we demonstrate the efficacy of our framework.

## Prerequisites

```
Matlab
vlfeat toolbox
```
## Usage
```
cub.m
sun.m
awa1.m
awa2.m
```

## Dataset
* AWA2: [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/) 
* CUB: [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)    
* SUN: [SUN Attribute](https://cs.brown.edu/~gen/sunattributes.html)

Complete Datasets can be downloaded [here](https://drive.google.com/open?id=1o0uvjk0y3saLzaOT0dn4jMfV4EXthVcy). For more detail about train/test split please refer to our [paper](https://arxiv.org/pdf/1707.08040.pdf)

## Result
![res](https://github.com/vkverma01/Zero-Shot/blob/master/results.png)
Here GFZSL-Trans represents the result in the transductive.

## References
If you are using this work please refer the ECML-17 paper: 

```
@inproceedings{verma2017simple,
  title={A simple exponential family framework for zero-shot learning},
  author={Verma, Vinay Kumar and Rai, Piyush},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={792--808},
  year={2017},
  organization={Springer}
}
```

## License

Code are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
