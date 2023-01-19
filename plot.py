import numpy as np
import matplotlib.pyplot as plt
ours_index = [4, 0, 5, 1, 6, 2, 7, 3]
ours = np.load('ours_result.npz')
hmp_w = np.load('human_motion_prediction_with_pose_result.npz')
hmp_wo = np.load('human_motion_prediction_wo_pose_result.npz')
pvrnn = np.load('pvrnn_result.npz')
for i in range(8):
    ax = plt.subplot(2, 4, i+1)
    l1, = plt.plot(ours['pred'][ours_index[i], :, 0], ours['pred'][ours_index[i], :, 1], '-rv')
    l2, = plt.plot(hmp_wo['pred'][i, :, 0], hmp_wo['pred'][i, :, 1], '-bv')
    l3, = plt.plot(hmp_w['pred'][i, :, 0], hmp_w['pred'][i, :, 1], '-yv')
    l4, = plt.plot(pvrnn['gt'][i, :, 0], pvrnn['gt'][i, :, 1], '-cv')
    l5, = plt.plot(hmp_w['gt'][i, :, 0], hmp_w['gt'][i, :, 1], '-gv')
    plt.plot(ours['pred'][ours_index[i], 0, 0], ours['pred'][ours_index[i], 0, 1], "rs")
    plt.plot(hmp_wo['pred'][i, 0, 0], hmp_wo['pred'][i, 0, 1], '-bs')
    plt.plot(hmp_w['pred'][i, 0, 0], hmp_wo['pred'][i, 0, 1], '-ys')
    plt.plot(pvrnn['gt'][i, 0, 0], pvrnn['gt'][i, 0, 1], '-cs')
    plt.plot(hmp_w['gt'][i, 0, 0], hmp_wo['gt'][i, 0, 1], '-gs')
    plt.title("plot {}".format(i))
    ax.legend(handles=[l1,l2,l3,l4,l5],labels=['ours', 'zero-velocity', 'martinez2017', 'pvrnn', 'ground truth'], loc='best')

plt.suptitle('human walking trajectory prediction')
plt.show()
