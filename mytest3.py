#pitf
from functools import reduce
import tensorflow as tf
import numpy as np
import time
import unittest
from functools import reduce
import logging
import tensorD.base.ops as ops
from numpy.random import rand
# all tensor dtype is tf.float32


def generate(shape, rank):
    """
    Generate matrix randomly(use standard normal distribution) by given shape and rank.

    Parameters
    ----------
    shape: int
        2-dim tuple.
        First element in tuple is the number of rows of the matrix U.
        Second element in tuple is the number of rows of the matrix V.

    rank: int
        The rank of matrix.
        And it`s also the number of columns of matrix U and V.

    Returns
    -------
    u,v: the generated matrix U and V.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3,4])
    >>> rank = tf.constant(5)
    >>> U, V = pitf_ops.generate((3,4), 5)
    >>> tf.Session().run(U)
    >>> tf.Session().run(V)

    """
    u = tf.random_normal((shape[0], rank), name='random_normal_u')
    v = tf.random_normal((shape[1], rank), name='random_normal_v')
    return u, v

#----------------------------------------------------------------------
shape = tf.constant([3,4])
rank = tf.constant(5)
U, V = generate((3,4), 5)
print(tf.Session().run(U))
print("")
print(tf.Session().run(V))
#-----------------------------------------------------------------------
def centralization(mat):
    """
    This function makes matrix to be centralized.

    Parameters
    ----------
    mat:
    The uncentralized matrix

    Returns
    -------
    ctr_mat:
    The centralized matrix.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> mat = tf.random_normal((3,4))
    >>> ctrmat = pitf_ops.centralization(mat)
    >>> tf.Session().run(ctrmat)
    """
    shape = tf.shape(mat)
    tmp = tf.matmul(tf.ones((shape[0], shape[0]), dtype=tf.float32, name='ctr_ones'), mat, name='ctr_mul')/tf.cast(shape[0], dtype=mat.dtype)
    ctr_mat = tf.subtract(mat, tmp, name='ctf_sub')
    return ctr_mat

mat = tf.random_normal((2,2))
ctrmat = centralization(mat)
print("")
print(tf.Session().run(ctrmat))
print("")
print(tf.Session().run(mat))
#-------------------------------
mat = tf.random_normal((2,2))
ctrmat = centralization(mat)
print("")
print(tf.Session().run(ctrmat))
print("")
print(tf.Session().run(mat))
#------------------------------

mat = tf.random_normal((2,2))
ctrmat = centralization(mat)
print("")
print(tf.Session().run(ctrmat))
print("")
print(tf.Session().run(mat))
#-------------------------------
mat = tf.random_normal((2,2))
ctrmat = centralization(mat)
print("")
print(tf.Session().run(ctrmat))
print("")
print(tf.Session().run(mat))
#------------------------------
mat21=tf.ones([3,3], tf.float32)
ctrmat = centralization(mat21)
print("")
print(tf.Session().run(mat21))

print("")
print(tf.Session().run(ctrmat))
#-----------------------------









