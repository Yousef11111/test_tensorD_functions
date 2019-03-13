import tensorD.base.ops as ops
from tensorD.base.type import DTensor
import numpy as np
import tensorflow as tf
tensor = tf.constant(np.arange(24).reshape(3,4,2))
unfolded_matrix = ops.unfold(tensor, 2)
print(tf.Session().run(unfolded_matrix))
sess=tf.Session()
a=tf.constant(5)
print(sess.run(a))
sess.close()
