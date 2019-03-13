import tensorD
import numpy as np
M=np.matrix('2 3;4 5; 6 7')
print(M)
print(tensorD.base.ops._skip(M,1))
print(tensorD.base.ops._gen_perm(6,2))
print(tensorD.base.ops._gen_perm(8,4))

