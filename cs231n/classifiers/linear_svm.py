import numpy as np
from random import shuffle



def svm_loss_naive(W, X, y, reg, dograd = False, verbose = False):
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
  - dograd: perform gradient descend

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """


  tM = np.zeros(W.shape) 
  dW = np.zeros(W.shape) # initialize the gradient as zero
  reg_mat_h = np.zeros(W.shape)
  reg_mat_mh = np.zeros(W.shape)

  step_size = 1e-5


  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train   = X.shape[0]

  loss = 0.0
  loss_grad_h    =   np.zeros(W.shape)   # plus  step_size
  loss_grad_mh   =   np.zeros(W.shape)   # minus step_size

  scores_grad  =   np.zeros([W.shape[0],W.shape[1]],dtype=object)
  correct_class_score  =   np.zeros(W.shape)
  correct_class_grad   =   np.zeros([W.shape[0],W.shape[1]])


  for i in xrange(num_train):
  #for i in xrange (1):

    scores = X[i].dot(W)
    if verbose: print "scores:", scores
    correct_class_score = scores[y[i]]

    if verbose: print "corr score:", correct_class_score


    d_grad_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      tM[i,j] = margin

      if verbose: print margin

      if margin > 0:
        loss += margin
        d_grad_count += 1

        # print dW.shape
        # print X[i].shape
        dW[:,j] += X[i]


    dW[:,y[i]] -= d_grad_count * X[i]

    ##
    ## Numerical estimation of the gradient
    ##

    # if dograd == True:
    #   for k in xrange(W.shape[0]):
    #     for l in xrange(W.shape[1]):

    #         W_h  = np.copy(W)      # plus step_size
    #         W_mh = np.copy(W)      # minus step_size

    #         W_h  [k,l] += step_size
    #         W_mh [k,l] -= step_size

    #         reg_mat_h [k,l] = np.sum(W_h * W_h)     # matrix for regulazation
    #         reg_mat_mh [k,l] = np.sum(W_mh * W_mh)  # matrix for regulazation

    #         scores_grad_h  = X[i].dot(W_h)
    #         scores_grad_mh = X[i].dot(W_mh)

    #         ###

    #         correct_class_grad_h  = scores_grad_h [y[i]]
    #         correct_class_grad_mh = scores_grad_mh[y[i]]

    #         ###

    #         for j in xrange(num_classes):
    #           if j == y[i]:
    #             continue

    #           margin_grad_h  = scores_grad_h  [j] - correct_class_grad_h  + 1 # note delta = 1
    #           margin_grad_mh = scores_grad_mh [j] - correct_class_grad_mh + 1 # note delta = 1

    #           ###

    #           if margin_grad_h > 0:
    #             loss_grad_h  [k,l]  += margin_grad_h

    #           if margin_grad_mh > 0:
    #             loss_grad_mh [k,l] += margin_grad_mh


    #  if i % 100 == 0:
    #      print (i)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  loss_grad_h /= num_train
  loss_grad_mh /= num_train

  dW /= num_train

  dW += 0.5 * reg * 2 * W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  #loss_grad_h  += 0.5 * reg * reg_mat_h
  #loss_grad_mh += 0.5 * reg * reg_mat_mh

  #dW = (loss_grad_h - loss_grad_mh) / (2 * step_size)


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

  delta = 1.0
  loss_sum = 0.0
  loss = 0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores = X.dot(W)


  unit_mat =  np.ones([X.shape[0],W.shape[1]])

  # calculating  'correct_class_scores':
  c = range(0,X.shape[0])
  correct_class_score = np.identity(X.shape[0])
  correct_class_score[c,c] = scores[c,y[c]]


  correct_class_score = np.dot(correct_class_score,unit_mat)
  correct_class_score [c,y[c]] = 0

  loss = scores - correct_class_score + unit_mat * delta



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




  ###
  ###    grad for y_i <> j
  ###

  ## Counting non null elements in loss function
  loss_uni = np.ones(loss.shape)
  loss_uni[loss<0] = 0

  dW = np.dot(np.transpose(X),loss_uni)


  ###
  ###    grad for y_i == j
  ###

  gr_minus = np.sum(loss>0, axis=1)

  zero_mat =  np.zeros([X.shape[0],W.shape[1]])
  zero_mat[c,y]=1


  gr_minus =  np.dot(np.identity(gr_minus.shape[0]) * gr_minus, zero_mat)

  dW -= np.dot(np.transpose(X), gr_minus)
  dW /= X.shape[0]
  dW += 0.5 * reg * 2 * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  loss[loss<0] = 0
  loss[c,y[c]] = 0

  loss_sum = (np.sum(loss)/X.shape[0]) + 0.5 * reg * np.sum(W * W)

  return loss_sum, dW




def old_vect_grad_desc():

      ## Old version of gradient descent

    from cs231n.classifiers.linear_svm import svm_loss_naive
    from cs231n.classifiers.linear_svm import svm_loss_vectorized

    loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.5, dograd = False)
    loss2, grad = svm_loss_vectorized(W, X_dev, y_dev, 0.5)

    print "loss: ", loss
    print "loss2: ", loss2


    # data
    #reg = 10000
    X = X_dev
    y = y_dev
    delta = 1
    loss_sum = 0.0
    loss = 0
    reg = 0.5
    step_size = 1e-5
    dW = np.zeros(W.shape) # initialize the gradient as zero
    dW2 = np.zeros(W.shape) # initialize the gradient as zero


    scores = X.dot(W)

    # calculating  'correct_class_scores':
    unit_mat =  np.ones([X.shape[0],W.shape[1]])

    c = range(0,X.shape[0])
    correct_class_score = np.identity(X.shape[0])
    correct_class_score[c,c] = scores[c,y[c]]


    correct_class_score = np.dot(correct_class_score,unit_mat)
    correct_class_score [c,y[c]] = 0

    loss = scores - correct_class_score + unit_mat * delta

    print loss.shape
    loss_uni = np.ones(loss.shape)
    loss_uni[loss<0] = 0
    print np.sum(loss_uni, axis=1)
    print loss_uni

    ## grad for y_i <> j
    dW2 = np.dot(np.transpose(X),loss_uni)
    print "dw2:",dW2.shape

    gr_minus = np.sum(loss>0, axis=1)
    print "gr_minus", gr_minus


    zero_mat =  np.zeros([X.shape[0],W.shape[1]])
    zero_mat[c,y]=1


    gr_minus =  np.dot(np.identity(gr_minus.shape[0]) * gr_minus, zero_mat)
    print gr_minus.shape
    print gr_minus[0:20,0:20]

    print "x shape: ", X.shape


    dW2 -= np.dot(np.transpose(X), gr_minus)
    dW2 /= X.shape[0]
    dW2 += 0.5 * reg * 2 * W

    print "loss matr dimensions: ", loss.shape
    print X.shape
    k = 1


    for k in xrange(X.shape[1]):
        for l in xrange (W.shape[1]):
            
            X_h = np.zeros ([X.shape[0],W.shape[1]])
            X_h[:,l] = X[:,k] * step_size
            
            
            c = range(0,X.shape[0])
            unit_mat =  np.ones([X.shape[0],W.shape[1]])
            correct_class_score = np.identity(X.shape[0])
            correct_class_score[c,c] = X_h[c,y[c]]
            correct_class_score = np.dot(correct_class_score,unit_mat)
            correct_class_score [c,y[c]] = 0
            

            loss_h = loss + X_h - correct_class_score
            loss_h[loss_h<0] = 0
            loss_h[c,y[c]] = 0
            


            loss_ph_sum = (np.sum(loss_h)/X.shape[0]) 
            
            
            loss_h = loss - X_h + correct_class_score
            loss_h[loss_h<0] = 0
            loss_h[c,y[c]] = 0
            loss_mh_sum = (np.sum(loss_h)/X.shape[0])
            

            dW[k,l] = loss_ph_sum - loss_mh_sum 
            
            
    dW = (dW + reg * 2 * W * step_size + step_size ** 2) / (2 * step_size)

    print dW.shape
    print dW[0,0:10]

    print "dW2"
    print dW2.shape
    print dW2[0,0:10]
        
    loss[loss<0] = 0
    loss[c,y[c]] = 0    
        
    difference = np.linalg.norm(dW - dW2, ord='fro')
    print ("diff ",difference)

    loss_sum = (np.sum(loss)/X.shape[0])  + 0.5 * reg * np.sum(W * W)

    print loss_sum
