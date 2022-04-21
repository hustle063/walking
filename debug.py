import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloader import NpzLoader
from utils.quaternion import forward_kinematics
from utils.visualization import render_animation
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    model = torch.load('E:/trained_model/checkpoint29_min_500.t7')
    file = 'h36m_test.npz'
    npzloader = NpzLoader(file, visualize=False, window=45)
    data = DataLoader(npzloader, batch_size=2, shuffle=False)

    for i_batch, sampled_batch in tqdm(enumerate(data)):
        if i_batch >= 6:
            break
        local_q = sampled_batch['local_q'].cuda()
        root_v = sampled_batch['xz_v'].cuda()
        worldpos = sampled_batch['worldpos'].cuda()
        root_pos = worldpos[:, :, 0:3]
        # root_pos = torch.index_select(root_pos, 2, torch.LongTensor([1, 2, 0]).cuda())
        std = torch.std(worldpos, axis=(0, 1), keepdim=True)
        debug = local_q.reshape(local_q.shape[0], local_q.shape[1], -1, 4)
        # debug = torch.index_select(debug, 3, torch.LongTensor([0, 3, 2, 1]).cuda())

        pos_pred = forward_kinematics(debug, root_pos, npzloader.data['rig'], npzloader.data['edges'])

        pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1, 3)

        pos_pred = pos_pred.cpu().detach().numpy()
        worldpos = worldpos.reshape(worldpos.shape[0], worldpos.shape[1], -1, 3)
        worldpos = worldpos.cpu().detach().numpy()
        # pos_pred[:, :, :, [2, 1, 0]] = pos_pred[:, :, :, [0, 1, 2]]
        # worldpos[:, :, :, [0, 2, 1]] = worldpos[:, :, :, [0, 1, 2]]
        for i in range(worldpos.shape[0]):
            sample = {'trajectory': worldpos[i], 'edges': npzloader.data['edges']}
            filename = 'walking_result/walk_predict_{}_{}.mp4'.format(i_batch, i)
            render_animation(sample, filename, fps=25)

