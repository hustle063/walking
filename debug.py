import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloader import NpzLoader
from utils.quaternion import forward_kinematics
from utils.visualization import render_animation
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    model = torch.load('trained_model/checkpoint10.t7')
    file = 'edin_test_less.npz'
    npzloader = NpzLoader(file, visualize=False)
    data = DataLoader(npzloader, batch_size=2, shuffle=True)

    for i_batch, sampled_batch in tqdm(enumerate(data)):
        if i_batch >= 6:
            break
        local_q = sampled_batch['local_q'].cuda()
        root_v = sampled_batch['root_v'].cuda()
        worldpos = sampled_batch['worldpos'].cuda()
        root_pos = worldpos[:, :, 0:3]
        std = torch.std(worldpos, axis=(0, 1), keepdim=True)
        debug = local_q.reshape(local_q.shape[0], local_q.shape[1], -1, 4)

        pos_pred = forward_kinematics(local_q.reshape(local_q.shape[0], local_q.shape[1], -1, 4),
                                      root_pos, npzloader.data['rig'], npzloader.data['edges'])

        pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1, 3)

        pos_pred = pos_pred.cpu().detach().numpy()
        for i in range(pos_pred.shape[0]):
            sample = {'trajectory': pos_pred[i], 'edges': npzloader.data['edges']}
            filename = 'walking_result/walk{}_{}.mp4'.format(i_batch, i)
            render_animation(sample, filename, fps=120)

