import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  for i in xrange(num_train):
      score = np.dot(X[i,:],W)
      score -= np.max(score)
      loss += -score[y[i]] + np.log(np.sum(np.exp(score)))
      for j in xrange(num_class):
          p = np.exp(score[j])/np.sum(np.exp(score))
          if j == y[i]:
              dW[:,j] += (p - 1)*X[i,:]
              continue

          dW[:,j] += p*X[i,:]


  loss /=num_train
  loss += reg*np.sum(np.sum(W*W))
  dW /=num_train
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = np.dot(X,W)
  scores -= np.matrix(np.max(scores,1)).T
  inv_sum_exp = 1./np.sum(np.exp(scores),1)
  p_mat = np.multiply(np.exp(scores),np.matrix(inv_sum_exp).T)

  loss_mat = -scores[xrange(X.shape[0]),y] + np.log(np.sum(np.exp(scores),1))
  loss = np.sum(np.matrix(loss_mat))
  loss += reg*np.sum(np.sum(W * W))
  loss /=X.shape[0]

  p_mat[xrange(X.shape[0]),y] -= 1
  dW = np.dot(X.T,p_mat)
  dW /=X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
