'''
Yichen edited based on Akshay's code
March, 2023

Code for the final Figure 1 submitted to CNS conference.
'''
import numpy as np
import matplotlib.pyplot as plt

from stimuli import StimulusGenerator
from area import Area
from brain import Brain
from utils import *

title_str = 'Skip Reciprocal Project. Similarity between singular vectors and input/output assemblies'
title_str += "\nN=1000, k=100, p=0.1, beta=0.1, norm_every=5, recurr=True,"+\
                "\nlearning max_iter=50, noise q=0.01, r=0.9, test_iter=5."
outfilename = 'CNS_Fig1.pdf'
print("Generating plot {}, parameters: {}".format(outfilename, title_str))

B = Brain(num_areas=3, num_neurons=1000, cap_size=100, p=0.1, beta=0.1)
# 3 regions: region X (input region), region Y or C (context region), region Z or V (variable region)

coreset_1 = np.arange(B.cap_size)
coreset_2 = np.arange(3*B.cap_size, 4*B.cap_size) # class chunking
S1 = StimulusGenerator(B.num_neurons, B.cap_size, background_p=0.01, coreset=coreset_1)
S2 = StimulusGenerator(B.num_neurons, B.cap_size, background_p=0.01, coreset=coreset_2)

y1, z2, W_yy1, W_zz1, W_xy1, W_zy1, W_xz1, W_yz1 = B.skip_reciprocal_project_stream(S1,0,1,2, max_iterations=50, return_weights_assembly=True)
y2, z2, W_yy2, W_zz2, W_xy2, W_zy2, W_xz2, W_yz2 = B.skip_reciprocal_project_stream(S2,0,1,2, max_iterations=50, initialize=False, return_weights_assembly=True)

X1 = S1.sample_stimuli(100)
X2 = S2.sample_stimuli(100)

# get a sample assembly using the first test input
y_new_1, z_new_1 = B.create_assemblies_skip_reciprocal_project(X1[:,0],0,1,2)
y_new_2, z_new_2 = B.create_assemblies_skip_reciprocal_project(X2[:,0],0,1,2)
# noiseless input
x1 = np.zeros(B.num_neurons)
x1[coreset_1] = 1.
x2 = np.zeros(B.num_neurons)
x2[coreset_2] = 1.
print("dist y1 - y2:", hamming(y_new_1, y_new_2))
print("dist z1 - z2:", hamming(z_new_1, z_new_2))
print("dist x1 - x2:", hamming(x1, x2))

print("calculating svd and distances")
Uxy,Sxy,Vxyt = np.linalg.svd(W_xy2)
Uzy,Szy,Vzyt = np.linalg.svd(W_zy2)
Uxz,Sxz,Vxzt = np.linalg.svd(W_xz2)
Uyz,Syz,Vyzt = np.linalg.svd(W_zy2)
Uyr,Syr,Vyrt = np.linalg.svd(W_yy2)
Uzr,Szr,Vzrt = np.linalg.svd(W_zz2)

Wxy_v_dist1 = np.array([hamming(x1, capk(np.absolute(Vxyt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wxy_u_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Uxy[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wzy_v_dist1 = np.array([hamming(z_new_1, capk(np.absolute(Vzyt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wzy_u_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Uzy[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wxz_v_dist1 = np.array([hamming(x1, capk(np.absolute(Vxzt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wxz_u_dist1 = np.array([hamming(z_new_1, capk(np.absolute(Uxz[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wyz_v_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Vyzt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wyz_u_dist1 = np.array([hamming(z_new_1, capk(np.absolute(Uyz[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wyy_v_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Vyrt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wyy_u_dist1 = np.array([hamming(y_new_1, capk(np.absolute(Uyr[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wzz_v_dist1 = np.array([hamming(z_new_1, capk(np.absolute(Vzrt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wzz_u_dist1 = np.array([hamming(z_new_1, capk(np.absolute(Uzr[:,i]), B.cap_size)) for i in range(B.num_neurons)])

Wxy_v_dist2 = np.array([hamming(x2, capk(np.absolute(Vxyt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wxy_u_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Uxy[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wzy_v_dist2 = np.array([hamming(z_new_2, capk(np.absolute(Vzyt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wzy_u_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Uzy[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wxz_v_dist2 = np.array([hamming(x2, capk(np.absolute(Vxzt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wxz_u_dist2 = np.array([hamming(z_new_2, capk(np.absolute(Uxz[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wyz_v_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Vyzt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wyz_u_dist2 = np.array([hamming(z_new_2, capk(np.absolute(Uyz[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wyy_v_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Vyrt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wyy_u_dist2 = np.array([hamming(y_new_2, capk(np.absolute(Uyr[:,i]), B.cap_size)) for i in range(B.num_neurons)])
Wzz_v_dist2 = np.array([hamming(z_new_2, capk(np.absolute(Vzrt[i]), B.cap_size)) for i in range(B.num_neurons)])
Wzz_u_dist2 = np.array([hamming(z_new_2, capk(np.absolute(Uzr[:,i]), B.cap_size)) for i in range(B.num_neurons)])

# sorting indices
Wxy_v_idx1 = np.argsort(Wxy_v_dist1)
Wxy_v_idx2 = np.argsort(Wxy_v_dist2)
Wxy_v_sort_order = np.concatenate([Wxy_v_idx1[:B.cap_size//2], Wxy_v_idx2[:B.cap_size//2], np.setdiff1d(Wxy_v_idx1[B.cap_size//2:], Wxy_v_idx2[:B.cap_size//2])])
Wxy_u_idx1 = np.argsort(Wxy_u_dist1)
Wxy_u_idx2 = np.argsort(Wxy_u_dist2)
Wxy_u_sort_order = np.concatenate([Wxy_u_idx1[:B.cap_size//2], Wxy_u_idx2[:B.cap_size//2], np.setdiff1d(Wxy_u_idx1[B.cap_size//2:], Wxy_u_idx2[:B.cap_size//2])])

Wzy_v_idx1 = np.argsort(Wzy_v_dist1)
Wzy_v_idx2 = np.argsort(Wzy_v_dist2)
Wzy_v_sort_order = np.concatenate([Wzy_v_idx1[:B.cap_size//2], Wzy_v_idx2[:B.cap_size//2], np.setdiff1d(Wzy_v_idx1[B.cap_size//2:], Wzy_v_idx2[:B.cap_size//2])])
Wzy_u_idx1 = np.argsort(Wzy_u_dist1)
Wzy_u_idx2 = np.argsort(Wzy_u_dist2)
Wzy_u_sort_order = np.concatenate([Wzy_u_idx1[:B.cap_size//2], Wzy_u_idx2[:B.cap_size//2], np.setdiff1d(Wzy_u_idx1[B.cap_size//2:], Wzy_u_idx2[:B.cap_size//2])])

Wxz_v_idx1 = np.argsort(Wxz_v_dist1)
Wxz_v_idx2 = np.argsort(Wxz_v_dist2)
Wxz_v_sort_order = np.concatenate([Wxz_v_idx1[:B.cap_size//2], Wxz_v_idx2[:B.cap_size//2], np.setdiff1d(Wxz_v_idx1[B.cap_size//2:], Wxz_v_idx2[:B.cap_size//2])])
Wxz_u_idx1 = np.argsort(Wxz_u_dist1)
Wxz_u_idx2 = np.argsort(Wxz_u_dist2)
Wxz_u_sort_order = np.concatenate([Wxz_u_idx1[:B.cap_size//2], Wxz_u_idx2[:B.cap_size//2], np.setdiff1d(Wxz_u_idx1[B.cap_size//2:], Wxz_u_idx2[:B.cap_size//2])])

Wyz_v_idx1 = np.argsort(Wyz_v_dist1)
Wyz_v_idx2 = np.argsort(Wyz_v_dist2)
Wyz_v_sort_order = np.concatenate([Wyz_v_idx1[:B.cap_size//2], Wyz_v_idx2[:B.cap_size//2], np.setdiff1d(Wyz_v_idx1[B.cap_size//2:], Wyz_v_idx2[:B.cap_size//2])])
Wyz_u_idx1 = np.argsort(Wyz_u_dist1)
Wyz_u_idx2 = np.argsort(Wyz_u_dist2)
Wyz_u_sort_order = np.concatenate([Wyz_u_idx1[:B.cap_size//2], Wyz_u_idx2[:B.cap_size//2], np.setdiff1d(Wyz_u_idx1[B.cap_size//2:], Wyz_u_idx2[:B.cap_size//2])])

Wyy_v_idx1 = np.argsort(Wyy_v_dist1)
Wyy_v_idx2 = np.argsort(Wyy_v_dist2)
Wyy_v_sort_order = np.concatenate([Wyy_v_idx1[:B.cap_size//2], Wyy_v_idx2[:B.cap_size//2], np.setdiff1d(Wyy_v_idx1[B.cap_size//2:], Wyy_v_idx2[:B.cap_size//2])])
Wyy_u_idx1 = np.argsort(Wyy_u_dist1)
Wyy_u_idx2 = np.argsort(Wyy_u_dist2)
Wyy_u_sort_order = np.concatenate([Wyy_u_idx1[:B.cap_size//2], Wyy_u_idx2[:B.cap_size//2], np.setdiff1d(Wyy_u_idx1[B.cap_size//2:], Wyy_u_idx2[:B.cap_size//2])])

Wzz_v_idx1 = np.argsort(Wzz_v_dist1)
Wzz_v_idx2 = np.argsort(Wzz_v_dist2)
Wzz_v_sort_order = np.concatenate([Wzz_v_idx1[:B.cap_size//2], Wzz_v_idx2[:B.cap_size//2], np.setdiff1d(Wzz_v_idx1[B.cap_size//2:], Wzz_v_idx2[:B.cap_size//2])])
Wzz_u_idx1 = np.argsort(Wzz_u_dist1)
Wzz_u_idx2 = np.argsort(Wzz_u_dist2)
Wzz_u_sort_order = np.concatenate([Wzz_u_idx1[:B.cap_size//2], Wzz_u_idx2[:B.cap_size//2], np.setdiff1d(Wzz_u_idx1[B.cap_size//2:], Wzz_u_idx2[:B.cap_size//2])])

sv_idxs = np.arange(1, B.num_neurons+1)




print("plotting...")
plt_limit = 400
FONTSIZE = 22
fig, ((ax5, ax6), (ax7, ax8), (ax3, ax4), (ax11, ax12)) = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(18,16))
fig.suptitle(title_str, fontsize=FONTSIZE)

ax5.plot(sv_idxs[:plt_limit], Wxz_v_dist1[Wxz_v_sort_order[:plt_limit]], color='#D62628', ls='--', label='Distance to Class 1')
ax5.plot(sv_idxs[:plt_limit], Wxz_v_dist2[Wxz_v_sort_order[:plt_limit]], color='#1E76B4', ls=':', label='Distance to Class 2')
ax5.legend(loc='lower right', fontsize=FONTSIZE-2)
ax5.set_ylabel('$W_{XV}$', fontsize=FONTSIZE)
ax5.set_title('$d_H(right\ v, x)$', fontsize=FONTSIZE)
ax6.plot(sv_idxs[:plt_limit], Wxz_u_dist1[Wxz_u_sort_order[:plt_limit]], color='#D62628', ls='--')
ax6.plot(sv_idxs[:plt_limit], Wxz_u_dist2[Wxz_u_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax6.set_title('$d_H(left\ u, v)$', fontsize=FONTSIZE)

ax7.plot(sv_idxs[:plt_limit], Wyz_v_dist1[Wyz_v_sort_order[:plt_limit]], color='#D62628', ls='--')
ax7.plot(sv_idxs[:plt_limit], Wyz_v_dist2[Wyz_v_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax7.set_ylabel('$W_{CV}$', fontsize=FONTSIZE)
ax7.set_title('$d_H(right\ v, c)$', fontsize=FONTSIZE)
ax8.plot(sv_idxs[:plt_limit], Wyz_u_dist1[Wyz_u_sort_order[:plt_limit]], color='#D62628', ls='--')
ax8.plot(sv_idxs[:plt_limit], Wyz_u_dist2[Wyz_u_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax8.set_title('$d_H(left\ u, v)$', fontsize=FONTSIZE)

ax3.plot(sv_idxs[:plt_limit], Wzy_v_dist1[Wzy_v_sort_order[:plt_limit]], color='#D62628', ls='--')
ax3.plot(sv_idxs[:plt_limit], Wzy_v_dist2[Wzy_v_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax3.set_ylabel('$W_{VC}$', fontsize=FONTSIZE)
ax3.set_title('$d_H(right\ v, v)$', fontsize=FONTSIZE)
ax4.plot(sv_idxs[:plt_limit], Wzy_u_dist1[Wzy_u_sort_order[:plt_limit]], color='#D62628', ls='--')
ax4.plot(sv_idxs[:plt_limit], Wzy_u_dist2[Wzy_u_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax4.set_title('$d_H(left\ u, c)$', fontsize=FONTSIZE)

ax11.plot(sv_idxs[:plt_limit], Wzz_v_dist1[Wzz_v_sort_order[:plt_limit]], color='#D62628', ls='--')
ax11.plot(sv_idxs[:plt_limit], Wzz_v_dist2[Wzz_v_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax11.set_ylabel('$W_{VV}$', fontsize=FONTSIZE)
ax11.set_title('$d_H(right\ v, v)$', fontsize=FONTSIZE)
ax11.set_xlabel('Singular Vector Index', fontsize=FONTSIZE)
ax12.plot(sv_idxs[:plt_limit], Wzz_u_dist1[Wzz_u_sort_order[:plt_limit]], color='#D62628', ls='--')
ax12.plot(sv_idxs[:plt_limit], Wzz_u_dist2[Wzz_u_sort_order[:plt_limit]], color='#1E76B4', ls=':')
ax12.set_title('$d_H(left\ u, v)$', fontsize=FONTSIZE)
ax12.set_xlabel('Singular Vector Index', fontsize=FONTSIZE)

for ax in fig.get_axes():
    ax.label_outer() # remove inner labels
    ax.tick_params(axis='both', labelsize=FONTSIZE-2) # set tick label font size

# plt.show()
plt.savefig(outfilename)
print("done")

