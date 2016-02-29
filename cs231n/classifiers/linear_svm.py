import numpy as np
from random import shuffle



def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  dW = np.zeros(W.shape) # initialize the gradient as zero
  W_h = np.zeros(W.shape)

  step_size_log = -10
  step_size = 10 ** step_size_log


  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  loss_grad    =   np.zeros(W.shape)
  scores_grad  =   np.zeros([W.shape[0],W.shape[1]],dtype=object)
  correct_class_score  =   np.zeros(W.shape)
  correct_class_grad   =   np.zeros([W.shape[0],W.shape[1]])
  margin_grad  =   np.zeros(W.shape)

  for i in xrange(num_train):
    


    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin



    for k in xrange(W.shape[0]):
      for l in xrange(W.shape[1]):
          W_h = W
          W_h [k,l] += step_size
          scores_grad [k,l] = np.array(X[i].dot(W_h))
          correct_class_grad [k,l] = scores_grad[k,l][y[i]]


          for j in xrange(num_classes):
            if j == y[i]:
              continue


          margin_grad [k][l] = scores_grad [k,l][j] - correct_class_grad [k,l] + 1 # note delta = 1
          if margin > 0:
          loss_grad[k,l] += margin_grad[k][l]


    if i % 100 == 0:
      print (i)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss_grad /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  loss_grad += 0.5 * reg * np.sum((W+step_size) * (W+step_size))

  dW = (loss_grad - loss) / step_size


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  delta = 1.0
  scores = W.dot(X)

  margins = np.maximum(0, scores - scores[y] + delta)
  margins[y] = 0
  loss_i = np.sum(margins)

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
