from functools import reduce
import tensorflow as tf
import numpy as np
import time
import unittest
from functools import reduce
import logging
import tensorD.base.ops as ops
from numpy.random import rand

import tensorD.base.pitf_ops as pit_op
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


def subspace(shape, rank, mode=None):
    """
    Make the matrix A,B,C from pairwise interaction tensor satisfy the constraints and uniqueness.

    Parameters
    ----------
    shape: int 2-dim tuple.
    First element in tuple is the number of rows of the matrix U.
    Second element in tuple is the number of rows of the matrix V.

    rank: int
    The rank of matrix.
    And it`s also the number of columns of matrix U and V.

    mode: str
    Point out to use function for which matrix.
    mode option is 'A', 'B','C'.(default option is None.)

    Returns
    -------
    Psb,Psc:
    The matrix which satisfied the constraints.

    Psa:
    The matrix which satisfied the constraints.But it`s more complicated than Psb and Psc
    because of the constraints difference.

    Examples
    --------
    >>> import tensorD.base.pitf_ops as ops
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> shape = tf.constant([3,4])
    >>> rank = tf.constant(3)
    >>> result = pitf_ops.subspace(shape,rank,'A')
    >>> tf.Session().run(result)
    """
    U, V = generate(shape, rank)
    tmp = tf.matmul(U, V, transpose_a=False, transpose_b=True,name='subspace_mul')
    if mode == 'B':
        Psb = centralization(tmp)
        return Psb
    if mode == 'C':
        Psc = centralization(tmp)
        return Psc
    if mode == 'A':
        row = shape[0]
        col = shape[1]
        vec1 = tf.ones((row, 1))
        vec1_t = tf.transpose(vec1)
        vec2 = tf.ones((col, 1))
        vec2_t = tf.transpose(vec2)
        Psa = centralization(tmp)+tf.matmul(tf.matmul(vec1_t, tmp), vec2)*(vec1*vec2_t)/tf.cast((row*col), dtype=tmp.dtype)
        return Psa
    return False
#--------------------------------------------------------------------------------------------------
shape = tf.constant([3,4])
rank = tf.constant(3)
result = pit_op.subspace(shape,rank,'A')
print(tf.Session().run(result))











    
