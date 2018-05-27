# mnist_cnn_autoencoder
Visualizing the learnt filters of a CNN trained for MNIST digit classification

Convolutional Neural Networks have been proved to successful in learning patterns in images efficiently,
and have an advantage over fully connected neural networks when it comes to image classification.

Each filter learns certain patterns which help it in identifying images. As we go deeper into the network,
we can observe that the filter learn more abstract (or higher level features). This is explained by Zeiler et al.
in Visualizing and Understanding Convolutional Networks [https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf]

Here, a CNN with one layer of convolutional filters (64 filters of size 5x5) is trained on the MNIST and the model
is saved(mnist_autoencoder.py). The model is later loaded and for test images, the outputs of the filters are visualized.
Bright colours in the result image show that those parts were noted by that filter (due to correlation between
the filter and the input image).

This is the link to the results. Ping me if the link is inaccessible

https://drive.google.com/open?id=1JNf58mXRmq9-U1j0RySHeuUTN64nr3rY

Files named <digit y>_<some number x>.png show what each filter has observed for the  input digit y.
Files named <digit>_<some number x>_true.png show the input image.
Files named <digit>_<some number x>_added.png show the accumulated filter response.
  
File mnist_autoencoder.py has to be run first as python mnist_autoencoder.py
File reading_models.py has to be run next as python reading_models.py [--cmap grey]

- Gautam S
  gautamsree@gmail.com
