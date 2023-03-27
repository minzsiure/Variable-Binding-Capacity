import numpy as np
import matplotlib.pyplot as plt

from stimuli import StimulusGenerator
from area import Area
from brain import Brain
from utils import *


B = Brain(num_areas=2, num_neurons=1000, cap_size=100, p=0.1, beta=0.1)

coreset_1 = np.arange(B.cap_size)
coreset_2 = np.arange(3*B.cap_size, 4*B.cap_size)
S1 = StimulusGenerator(B.num_neurons, B.cap_size, background_p=0.01, coreset=coreset_1)
S2 = StimulusGenerator(B.num_neurons, B.cap_size, background_p=0.01, coreset=coreset_2)

y1, W_yy1, W_yx1 = B.project_stream(S1,0,1, max_iterations=50, return_weights_assembly=True)
y2, W_yy2, W_yx2 = B.project_stream(S2,0,1, max_iterations=50, initialize=False, return_weights_assembly=True)

X1 = S1.sample_stimuli(100)
X2 = S2.sample_stimuli(100)

# y_new_1 = capk(W_yx2.dot(X1[:,0]), B.cap_size)
# y_new_2 = capk(W_yx2.dot(X2[:,0]), B.cap_size)
y_new_1 = B.create_assemblies(X1[:, 0], 0,1)
y_new_2 = B.create_assemblies(X2[:, 0], 0,1)
print(hamming(y_new_1, y_new_2))

# W_yx /= W_yx.sum(axis=1)[:,np.newaxis]

x1 = np.zeros(B.num_neurons)
x1[coreset_1] = 1.
x2 = np.zeros(B.num_neurons)
x2[coreset_2] = 1.

U1,S1,Vt1 = np.linalg.svd(W_yx2)
Ur1,Sr1,Vrt1 = np.linalg.svd(W_yy2)

x_dist1 = np.array([hamming(x1, capk(np.absolute(Vt1[i]), B.cap_size)) for i in range(B.num_neurons)])
y_dist1 = np.array([hamming(y_new_1, capk(np.absolute(U1[:,i]), B.cap_size)) for i in range(B.num_neurons)])
x_dist2 = np.array([hamming(x2, capk(np.absolute(Vt1[i]), B.cap_size)) for i in range(B.num_neurons)])
y_dist2 = np.array([hamming(y_new_2, capk(np.absolute(U1[:,i]), B.cap_size)) for i in range(B.num_neurons)])

yrv_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Vrt1[i]), B.cap_size)) for i in range(B.num_neurons)])
yru_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Ur1[:,i]), B.cap_size)) for i in range(B.num_neurons)])
yrv_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Vrt1[i]), B.cap_size)) for i in range(B.num_neurons)])
yru_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Ur1[:,i]), B.cap_size)) for i in range(B.num_neurons)])

x_idxs1 = np.argsort(x_dist1)
x_idxs2 = np.argsort(x_dist2)
x_sort_order = np.concatenate([x_idxs1[:B.cap_size//2], x_idxs2[:B.cap_size//2], np.setdiff1d(x_idxs1[B.cap_size//2:], x_idxs2[:B.cap_size//2])])

y_idxs1 = np.argsort(y_dist1)
y_idxs2 = np.argsort(y_dist2)
y_sort_order = np.concatenate([y_idxs1[:B.cap_size//2], y_idxs2[:B.cap_size//2], np.setdiff1d(y_idxs1[B.cap_size//2:], y_idxs2[:B.cap_size//2])])

yrv_idxs1 = np.argsort(yrv_dist1)
yrv_idxs2 = np.argsort(yrv_dist2)
yrv_sort_order = np.concatenate([yrv_idxs1[:B.cap_size//2], yrv_idxs2[:B.cap_size//2], np.setdiff1d(yrv_idxs1[B.cap_size//2:], yrv_idxs2[:B.cap_size//2])])

yru_idxs1 = np.argsort(yru_dist1)
yru_idxs2 = np.argsort(yru_dist2)
yru_sort_order = np.concatenate([yru_idxs1[:B.cap_size//2], yru_idxs2[:B.cap_size//2], np.setdiff1d(yru_idxs1[B.cap_size//2:], yru_idxs2[:B.cap_size//2])])

sv_idxs = np.arange(1, B.num_neurons+1)

plt_limit = 400
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,6))
fig.suptitle('Similarity between singular vectors and input/output assemblies')
ax1.plot(sv_idxs[:plt_limit], x_dist1[x_sort_order[:plt_limit]], 'r--', label='Distance to Class 1')
ax1.plot(sv_idxs[:plt_limit], x_dist2[x_sort_order[:plt_limit]], 'b:', label='Distance to Class 2')
ax1.legend(loc='lower right')
ax1.set_ylabel('$W_{AS}$', fontsize=16)
ax1.set_title('$d_H(v, x)$')
ax2.plot(sv_idxs[:plt_limit], y_dist1[y_sort_order[:plt_limit]], 'r--')
ax2.plot(sv_idxs[:plt_limit], y_dist2[y_sort_order[:plt_limit]], 'b:')
ax2.set_title('$d_H(u, y)$')
ax3.plot(sv_idxs[:plt_limit], yrv_dist1[yrv_sort_order[:plt_limit]], 'r--')
ax3.plot(sv_idxs[:plt_limit], yrv_dist2[yrv_sort_order[:plt_limit]], 'b:')
ax3.set_ylabel('$W_{AA}$', fontsize=16)
ax3.set_xlabel('Singular Vector Index')
ax3.set_title('$d_H(v, y)$')
ax4.plot(sv_idxs[:plt_limit], yru_dist1[yru_sort_order[:plt_limit]], 'r--')
ax4.plot(sv_idxs[:plt_limit], yru_dist2[yru_sort_order[:plt_limit]], 'b:')
ax4.set_xlabel('Singular Vector Index')
ax4.set_title('$d_H(u, y)$')

for ax in fig.get_axes():
    ax.label_outer()

plt.show()

