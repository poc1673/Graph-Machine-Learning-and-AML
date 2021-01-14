# Machine Learning for Graphs

The purpose of this project is to work as my primer on machine learning in networks, with an emphasis on the application of these models for analyzing instances of money laundering or fraud in networks of transactions.

The focus of this project will be on academic literature and numerical experiments that have been implemented in literature in the past few years. Papers that are deemed as necessary precursors will also be reviewed.

Implementation will be in Python Tensorflow.

## Scope:

The scope will be a survey of each of the major methodologies used in machine learning for networks. At the time of this assessment (2020-12-19) this will be:

* Graph convolutional neural networks.
* Manifold propogation
* DeepWalk and skip-gram graph embeddings
* More simplistic methodologies such as the logistic regression framework used in earlier papers.

After reviewing and replicating the above results, a novel approach should be developed using a data-set. It should also be applied to data-sets related to fraud/money laundering networks. This novel implementation should begin with a motivation and derivation of the method, its performance, and a full explanation of how it works.

## Final Result:

1. Familiarity with major forms of machine learning for networks.
2. Original implementation of graph learning algorithm implemented.
3. Presentation of novel implementation compared to previous implementations using visualization and explanation.

## Timeline:

| Week # | Begin Date | Topic | Items Due |   
|------------|----------------------|-------|-----------|
| 1|  2021-01-10         |    Refresher     | Graph theory/Tensorflow review      |           |   
|2|  2021-02-02          |    Node embedding methods      |  Skip-gram graph embeddings (DeepWalk)     |           |   
| 3|  2021-01-18         |    GCN and neural network based methods      | Convolutional neural networks and graph theory      |           |   
| 4|  2021-01-26          |          |    Manifold propogation   |           |   
| 5|  2021-02-10          |          |   FAST CGN Replication    |           |   
|6|  2021-02-18          |          | Create novel model.       |           |   
|7 | 2021-02-26          |          | Generate writeup on results.       |           |

### Week 1:
**Review theory for neural networks:**
 + Basic principles of neural networks.
 + Structure of networks (RNN, et cetera).
 + Review of graph convolutional neural networks.

**Tensorflow refresher with exercises:**
 + Review graph structure, programming framework for TensorFlow.
 + Review backpropogation.
 + Tensorflow ML exercise #1: Solve unconstrained optimization problem.
 + Tensorflow ML exercise #2: Solve constrained optimization problem.
 + Tensorflow ML exercise #3: Fit linear regression from scratch, fit "two-stage" linear regression.
 + Employ simple machine learning on the graph structure to try to group data.

#### Resources:
* [node2vec: Embeddings for Graph Data
](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)
* [Stellargraph Example of Node2Vec](https://stellargraph.readthedocs.io/en/v1.0.0rc1/demos/node-classification/node2vec/stellargraph-node2vec-weighted-random-walks.html)
* [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
* [How to get started with machine learning on graphs](https://medium.com/octavian-ai/how-to-get-started-with-machine-learning-on-graphs-7f0795c83763)
* [Octavian Machine Learning on Graphs Course](https://octavian.ai/machine-learning-on-graphs-course.html)
* [Deep Learning with Python: Chapters 2 and 3](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?dchild=1&keywords=francois+chollet&qid=1608392039&sr=8-1)
* [Cora Data-set](https://www.google.com/search?q=cora+dataset&oq=cora+dataset&aqs=chrome..69i57j69i60l3j35i39j0.2535j0j1&sourceid=chrome&ie=UTF-8)


### Week 2:

#### Resources:
* [node2vec Website](http://snap.stanford.edu/node2vec/)
* [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
* [node2vec: Embeddings for Graph Data
](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
* [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
* [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
* [An Illustrated Explanation of Using SkipGram To Encode The Structure of A Graph (DeepWalk)
](https://medium.com/@_init_/an-illustrated-explanation-of-using-skipgram-to-encode-the-structure-of-a-graph-deepwalk-6220e304d71b#:~:text=DeepWalk%20is%20an%20algorithm%20that,community%20structure%20of%20the%20graph.&text=However%2C%20SkipGram%20is%20an%20algorithm,used%20to%20create%20word%20embeddings.)


### Week 3:

1. Read Semi-Supervised Classification on Graph Convolutional Networks - Kipf Welling - 2017.
2. Replicate their results using Tensorflow 2.0 (original implementation used earlier version).
3. (If possible) Implement custom functions for GCNN using Tensorflow.


### Week 4:





### Week 5:

* Read FASTCGN implementation paper.
* Implement changes from FASTCGN onto original GCNN functions to create fast versions.
* Attempt to replicate results from both papers.

### Week 6:

To reassess steps.


### Week 7:







Ending: 2021-02-28

## Resources

### Graph Theory

### Deep Learning
1. [Tensorflow 2.0 Website](https://www.tensorflow.org/guide/effective_tf2)
2. [Tensorflow cheat sheet](http://www.aicheatsheets.com/static/pdfs/tensorflow_v_2.0.pdf)
3. [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?dchild=1&keywords=francois+chollet&qid=1608392039&sr=8-1)
4. [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646/ref=sr_1_3?dchild=1&keywords=francois+chollet&qid=1608392039&sr=8-3)

## Data-sets
1. [Karate club data-set](http://networkrepository.com/soc-karate.php)
2. [Elliptic data-set](https://www.kaggle.com/ellipticco/elliptic-data-set)
# Graph-Machine-Learning-and-AML
