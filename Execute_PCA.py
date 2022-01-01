from PCA import *
from kernel_PCA import *

p = PCA('hw06-data1.mat')
# p = PCA('hw06-data2.mat')
p.run_algo()

k = kPCA('hw06-data1.mat')
# k = kPCA('hw06-data2.mat')
k.run_algo()
