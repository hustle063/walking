from models.network import PaceEncoder, ControlEncoder, PaceDecoder, GRUModel1, PaceBlock
from utils.quaternion import normalize_quaternion, forward_kinematics, geodesic_loss, qeuler, euler_to_quaternion
from utils.visualization import render_animation
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch
import torch.optim as optim
import numpy as np
import math
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm


class PaceNetwork:
    def __init__(self, input_dim=96):
        self.leftleg_layer = PaceBlock(in_dim=15)
        self.leftleg_layer = self.leftleg_layer.cuda()
        self.rightleg_layer = PaceBlock(in_dim=15)
        self.rightleg_layer = self.rightleg_layer.cuda()
        self.torso_layer = PaceBlock(in_dim=15)
        self.torso_layer = self.torso_layer.cuda()
        self.leftarm_layer = PaceBlock(in_dim=24)
        self.leftarm_layer = self.leftarm_layer.cuda()
        self.rightarm_layer = PaceBlock(in_dim=24)
        self.rightarm_layer = self.rightarm_layer.cuda()
        self.decoder = PaceDecoder(in_dim=162)
        self.decoder = self.decoder.cuda()
        self.optimizer_g = optim.Adam(lr=0.0001, params=list(self.leftleg_layer.parameters()) +
                                                        list(self.rightleg_layer.parameters()) +
                                                        list(self.leftarm_layer.parameters()) +
                                                        list(self.rightarm_layer.parameters()) +
                                                        list(self.torso_layer.parameters()) +
                                                        list(self.decoder.parameters()), betas=(0.5, 0.9))
        self.min_loss_pos_val_480 = np.inf
        self.min_loss_pos_val_320 = np.inf
        self.min_loss_pos_val_160 = np.inf
        self.min_loss_angle_560 = np.inf
        self.min_loss_angle_320 = np.inf
        self.min_loss_angle_160 = np.inf
        self.min_loss_angle_160 = np.inf
        self.local_q = None
        self.worldpos = None
        self.xz_v = None
        self.root_v = None
        self.root_p = None
        self.rot_angle = None
        self.face_dir = None
        self.xz_v_pred = None
        self.y_v_pred = None
        self.rot_angle_pred = None
        self.distance_pred = None
        self.rot_angle_norm = None
        self.local_q_next = None
        self.skeleton_scale = None
        self.h_leftleg = None
        self.h_rightleg = None
        self.h_torso = None
        self.h_leftarm = None
        self.h_rightarm = None
        self.h_gru = None
        self.h_gru1 = None

    def read_data(self, batch_data):
        self.local_q = batch_data['local_q'].cuda()
        self.root_v = batch_data['root_v'].cuda()
        self.root_p = batch_data['root_p'].cuda()
        self.worldpos = batch_data['worldpos'].cuda()
        self.xz_v = batch_data['xz_v'].cuda()
        self.face_dir = batch_data['face_dir'].cuda()
        self.rot_angle = batch_data['rot_angle'].cuda()
        self.skeleton_scale = batch_data['scale'].cuda()
        self.worldpos = self.skeleton_scale.unsqueeze(-1).unsqueeze(-1).expand(self.worldpos.shape[0],
                                                                               self.worldpos.shape[1],
                                                                               self.worldpos.shape[2]) * self.worldpos
        self.xz_v = self.skeleton_scale.unsqueeze(-1).expand(-1, self.xz_v.shape[1]) * self.xz_v
        return self.local_q.size()[0]

    def forward(self, t, easy_state=True):
        if easy_state:
            # local_q_t = self.local_q[:, t]
            rot_angle_t = self.rot_angle[:, t, :]
        else:
            # local_q_t = self.local_q[:, t]
            rot_angle_t = self.rot_angle_pred[0]
        leftleg_pos = self.worldpos[:, t, 3:18]
        rightleg_pos = self.worldpos[:, t, 18:33]
        torso_pos = self.worldpos[:, t, 33:48]
        leftarm_pos = self.worldpos[:, t, 48:72]
        rightarm_pos = self.worldpos[:, t, 72:96]
        x_leftleg, self.h_leftleg = self.leftleg_layer(torch.cat([leftleg_pos], -1).unsqueeze(0), self.h_leftleg)
        x_rightleg, self.h_rightleg = self.rightleg_layer(torch.cat([rightleg_pos], -1).unsqueeze(0), self.h_rightleg)
        x_torso, self.h_torso = self.torso_layer(torch.cat([torso_pos], -1).unsqueeze(0), self.h_torso)
        x_leftarm, self.h_leftarm = self.leftarm_layer(torch.cat([leftarm_pos], -1).unsqueeze(0), self.h_leftarm)
        x_rightarm, self.h_rightarm = self.rightarm_layer(torch.cat([rightarm_pos], -1).unsqueeze(0), self.h_rightarm)
        state_input = torch.cat([rot_angle_t], -1).unsqueeze(0)
        h_out = torch.cat([x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm, state_input], -1)
        self.rot_angle_pred, self.xz_v_pred = self.decoder(h_out)

    def train(self, npzloader_train, npzloader_val, n_epoch, batch_num, writer, window=25):
        torch.autograd.set_detect_anomaly(True)
        data = DataLoader(npzloader_train, batch_size=batch_num, shuffle=True)
        for epoch in range(n_epoch):
            # self.state_encoder.train()
            self.leftleg_layer.train()
            self.rightleg_layer.train()
            self.torso_layer.train()
            self.leftarm_layer.train()
            self.rightarm_layer.train()
            self.decoder.train()
            rig_pred = {}
            for i_batch, sampled_batch in tqdm(enumerate(data)):
                # loss_q = 0
                loss_xz = 0
                # loss_y = 0
                loss_rot = 0
                loss_distance = 0
                loss_L2 = torch.nn.MSELoss()
                # std = torch.std(self.worldpos, axis=(0, 1), keepdim=True)
                batch_size = self.read_data(sampled_batch)
                self.optimizer_g.zero_grad()
                self.h_leftleg = None
                self.h_rightleg = None
                self.h_torso = None
                self.h_leftarm = None
                self.h_rightarm = None
                stage = None
                for t in range(window - 1):
                    seed = np.random.uniform(0, 1)
                    if seed > (math.log10(epoch + 1) / 6 + 0.3) or t < 10 - epoch / 50:
                        stage = True
                    else:
                        stage = False
                    if t == 0:  # Initial start has to be in easy state
                        self.forward(t, easy_state=True)
                    elif stage:
                        self.forward(t, easy_state=True)
                    else:
                        self.forward(t, easy_state=False)
                    sin_angle = self.rot_angle[:, t + 1, 0].unsqueeze(0)
                    cos_angle = self.rot_angle[:, t + 1, 1].unsqueeze(0)
                    # new_sin_angle = math.sqrt(2) / 2 * (sin_angle+cos_angle)
                    # new_cos_angle = math.sqrt(2) / 2 * (cos_angle-sin_angle)

                    # xz_v_next = self.xz_v[:, t + 1:t + 2]

                    sin_angle_pred_1 = self.rot_angle_pred[:, :, 0]
                    cos_angle_pred_1 = self.rot_angle_pred[:, :, 1]

                    face_dir_1 = torch.clone(self.face_dir[:, t + 1, :])
                    dir_column1 = (torch.cat([cos_angle_pred_1, sin_angle_pred_1]).T * face_dir_1).sum(1, keepdim=True)
                    dir_column2 = (torch.cat([-sin_angle_pred_1, cos_angle_pred_1]).T * face_dir_1).sum(1, keepdim=True)

                    dir_column3 = (torch.cat([cos_angle, sin_angle]).T * face_dir_1).sum(1, keepdim=True)
                    dir_column4 = (torch.cat([-sin_angle, cos_angle]).T * face_dir_1).sum(1, keepdim=True)
                    xz_dir_pred = torch.cat([dir_column1, dir_column2], 1)
                    xz_dir_gt = torch.cat([dir_column3, dir_column4], 1)
                    xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1) / self.skeleton_scale
                    xz_v = xz_v_next.view(-1, 1).expand_as(xz_dir_pred)
                    xz_v_pred = self.xz_v_pred.view(-1, 1).expand_as(xz_dir_pred)
                    distance_pred = xz_v_pred * xz_dir_pred
                    distance_gt = xz_v * xz_dir_gt
                    # distance_gt = distance_gt * self.skeleton_scale.view(-1, 1).expand_as(distance_pred)
                    loss_distance += loss_L2(distance_pred[:, :], distance_gt)

                    if epoch % 10 == 0:
                        if t == 0:
                            with open('value3.txt', 'a') as file_object:
                                file_object.write('epoch {}:\n'.format(epoch))
                        # print(self.xz_v_pred[0].clone().detach().cpu().numpy(), xz_v_next)
                        with open('value3.txt', 'a') as file_object:
                            file_object.write('distance angle timestep {} stage {}\n'.format(t, stage))
                        with open('value3.txt', 'ab') as file_object:
                            sin = np.stack((distance_pred[0].clone().detach().cpu().numpy(),
                                            distance_gt[0].clone().cpu().numpy()))
                            np.savetxt(file_object, sin)
                        # with open('value.txt', 'a') as file_object:
                        # file_object.write('cos angle timestep {} stage {}'.format(t, stage))
                        # with open('value.txt', 'ab') as file_object:
                        # np.savetxt(file_object, self.rot_angle_pred[0, :, 1].clone().detach().cpu().numpy())
                        # np.savetxt(file_object, cos_angle.clone().cpu().numpy())

                    # pred_angle = torch.atan(torch.div(sin_angle_pred, cos_angle_pred))
                    # angle = torch.atan(torch.div(sin_angle, cos_angle))
                    # angle_loss1 = torch.abs(angle - pred_angle)
                    # angle_loss2 = torch.abs(torch.sub(angle_loss1, 2 * torch.tensor(math.pi)))

                    # self.rot_angle_pred[0, :, 0] = math.sqrt(2) / 2 * (sin_angle_pred - cos_angle_pred)
                    # self.rot_angle_pred[0, :, 1] = math.sqrt(2) / 2 * (sin_angle_pred + cos_angle_pred)

                    loss_rot += torch.mean(xz_v_next * (
                            torch.abs(sin_angle - sin_angle_pred_1) + torch.abs(cos_angle - cos_angle_pred_1)))

                loss_total = loss_distance / (window - 1)
                loss_total.backward(retain_graph=True)
                parameters = list(self.leftleg_layer.parameters()) + list(self.rightleg_layer.parameters()) + \
                             list(self.leftarm_layer.parameters()) + list(self.rightarm_layer.parameters()) + \
                             list(self.torso_layer.parameters()) + list(self.decoder.parameters())
                # clip_grad_norm_(parameters, 0.1)
                self.optimizer_g.step()

                # writer.add_scalar('loss_xz', loss_xz.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_rotation', loss_rot.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_distance', loss_distance.item(), global_step=epoch * len(data) + i_batch)
                # writer.add_scalar('loss_total', loss_total.item(), global_step=epoch * len(data) + i_batch)

            if epoch % 10 == 0:
                # print(self.xz_v_pred[0].clone().detach().cpu().numpy(), xz_v_next)
                print(self.rot_angle_pred[0, :, 1].clone().detach().cpu().numpy(), cos_angle)
                print(self.rot_angle_pred[0, :, 0].clone().detach().cpu().numpy(), sin_angle)
            if epoch % 1 == 0:
                self.validation(npzloader_val, data, epoch, writer)
        self.save_model('C:/Users/yuchhuang9/walking/trained_model/h36m_pace_33_final.t7', epoch)

    def validation(self, npzloader, data_train, epoch, writer, window=25):
        notsave = True
        data_val = DataLoader(npzloader, batch_size=32, drop_last=True)
        rig_val = {}
        errors = []
        loss_pos_val_160 = 0
        loss_pos_val_320 = 0
        loss_pos_val_480 = 0
        # self.state_encoder.eval()
        self.leftleg_layer.eval()
        self.rightleg_layer.eval()
        self.torso_layer.eval()
        self.leftarm_layer.eval()
        self.rightarm_layer.eval()
        self.decoder.eval()
        for j_batch, val_batch in tqdm(enumerate(data_val)):
            batch_size = self.read_data(val_batch)
            traj_point = self.root_p[:, 0, [0, 1]]
            loss_rot = 0
            self.hidden = None
            for t in range(window):  # window length
                if t == 0:
                    self.forward(t, easy_state=True)
                else:
                    self.forward(t, easy_state=False)
                sin_angle = self.rot_angle[:, t + 1, 0]
                cos_angle = self.rot_angle[:, t + 1, 1]

                xz_v_next = self.xz_v[:, t + 1:t + 2]
                sin_angle_pred = self.rot_angle_pred[0, :, 0]
                cos_angle_pred = self.rot_angle_pred[0, :, 1]
                # loss_xz += torch.mean(torch.abs(xz_v_next - self.xz_v_pred[0]).squeeze(-1))
                loss_rot += torch.mean(xz_v_next * (
                        torch.abs(sin_angle - sin_angle_pred) + torch.abs(cos_angle - cos_angle_pred)))

                # print(sin_angle, cos_angle)

            # loss_xz = loss_xz/(window-1)
            loss_rot = loss_rot / (window - 1)

        writer.add_scalar('loss_rot_val', loss_rot.item(), global_step=epoch * len(data_train))

    def save_model(self, filename, epoch):
        state = {
            'epoch': epoch,
            # 'state_encoder': self.state_encoder.state_dict(),
            # 'control_encoder': self.control_encoder.state_dict(),
            'leftleg_layer': self.leftleg_layer.state_dict(),
            'rightleg_layer': self.rightleg_layer.state_dict(),
            'torso_layer': self.torso_layer.state_dict(),
            'leftarm_layer': self.leftarm_layer.state_dict(),
            'rightarm_layer': self.rightarm_layer.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimize': self.optimizer_g.state_dict()
        }
        torch.save(state, filename)

    def predict(self, npzloader, model_file, batch_num, window=75):
        np.set_printoptions(suppress=True)
        data = DataLoader(npzloader, batch_size=batch_num, shuffle=False)
        model = torch.load(model_file)
        # self.state_encoder.load_state_dict(model['state_encoder'])
        # self.control_encoder.load_state_dict(model['control_encoder'])
        self.leftleg_layer.load_state_dict(model['leftleg_layer'])
        self.rightleg_layer.load_state_dict(model['rightleg_layer'])
        self.torso_layer.load_state_dict(model['torso_layer'])
        self.leftarm_layer.load_state_dict(model['leftarm_layer'])
        self.rightarm_layer.load_state_dict(model['rightarm_layer'])
        self.decoder.load_state_dict(model['decoder'])
        FDEs = []
        ADEs = []
        preds = []
        gts = []
        # self.state_encoder.eval()
        # self.control_encoder.eval()
        self.leftleg_layer.eval()
        self.rightleg_layer.eval()
        self.torso_layer.eval()
        self.leftarm_layer.eval()
        self.rightarm_layer.eval()
        self.decoder.eval()
        for i_batch, sampled_batch in tqdm(enumerate(data)):
            batch_size = self.read_data(sampled_batch)
            traj_point = self.root_p[:, window-26, [0, 1]]
            traj_point1 = self.root_p[:, window-26, [0, 1]]
            euler_pred_seq = np.zeros((batch_size, window - 1, 78))
            self.hidden = None
            ground_truth = []
            predicted = []

            ADE = 0

            for t in range(window - 1):  # window length
                if t < window-25:
                    self.forward(t, easy_state=True)
                else:
                    self.forward(t, easy_state=False)
                sample = 0
                sin_angle_pred = self.rot_angle_pred[0, :, 0]
                cos_angle_pred = self.rot_angle_pred[0, :, 1]
                sin_angle = self.rot_angle[:, t+1, 0]
                cos_angle = self.rot_angle[:, t+1, 1]
                rot = torch.stack([torch.stack([cos_angle, -sin_angle]),
                                   torch.stack([sin_angle, cos_angle])])

                face_dir = torch.clone(self.face_dir[:, t + 1, :]).unsqueeze(-1)
                xz_dir = torch.bmm(torch.movedim(rot, 2, 0), face_dir).squeeze(-1)
                xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1) / self.skeleton_scale
                # xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1)
                xz_v = xz_v_next.view(-1, 1).expand_as(xz_dir)
                distance = xz_v * xz_dir
                if t >= window-26:
                    traj_point1 = distance + traj_point1

                rot_pred = torch.stack([torch.stack([cos_angle_pred, -sin_angle_pred]),
                                        torch.stack([sin_angle_pred, cos_angle_pred])])
                xz_dir = torch.bmm(torch.movedim(rot_pred, 2, 0), face_dir).squeeze(-1)
                xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1) / self.skeleton_scale
                # xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1)
                xz_v_pred = self.xz_v_pred.view(-1, 1).expand_as(xz_dir)
                distance_pred = xz_v_pred * xz_dir
                if t >= window-26:
                    traj_point = distance_pred + traj_point
                    error = torch.sqrt(torch.sum(torch.pow(traj_point1 - traj_point, 2), 1))
                    error = torch.mean(error)
                    ADE += error
                    if t == window-2:
                        FDE = error



                    predicted.append(traj_point[sample, :].clone().detach().cpu().numpy())
                    ground_truth.append(traj_point1[sample, :].clone().detach().cpu().numpy())

                #ground_truth.append(self.root_p[sample, t + 1, [0, 2]].clone().cpu().numpy())

                # sin_angle_pred_1 = self.rot_angle_pred[:, :, 0]
                # cos_angle_pred_1 = self.rot_angle_pred[:, :, 1]
                # face_dir_1 = torch.clone(self.face_dir[:, t + 1, :])
                # dir_column1 = (torch.cat([cos_angle_pred_1, sin_angle_pred_1]).T * face_dir_1).sum(1, keepdim=True)
                # dir_column2 = (torch.cat([-sin_angle_pred_1, cos_angle_pred_1]).T * face_dir_1).sum(1, keepdim=True)
                # xz_dir = torch.cat([dir_column1, dir_column2], 1)
                # xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1) / self.skeleton_scale
                # xz_v_pred = xz_v_next.view(-1, 1).expand_as(xz_dir)
                # distance_pred = xz_v_pred * xz_dir
                # traj_point = distance_pred + traj_point
                # predicted.append(traj_point[sample, :].clone().detach().cpu().numpy())
                # ground_truth.append(self.root_p[sample, t + 1, [0, 2]].clone().cpu().numpy())



                # error = torch.mean(torch.sqrt(((traj_point - self.root_p[:, t + 1, [0, 2]]) ** 2).sum(-1)))
                # mean_errors = error.cpu().detach().numpy()
                # print(cos_angle_pred.clone().detach().cpu().numpy(), self.rot_angle[:, t + 1, 1])
                # print(sin_angle_pred.clone().detach().cpu().numpy(), self.rot_angle[:, t + 1, 0])
                # errors.append(mean_errors)
            ADE = ADE / 25
            FDEs.append(FDE.clone().detach().cpu().numpy())
            ADEs.append(ADE.clone().detach().cpu().numpy())

            # predicted.insert(0, self.root_p[sample, 0, [0, 1]].clone().cpu().numpy())
            # ground_truth.insert(0, self.root_p[sample, 0, [0, 1]].clone().cpu().numpy())
            pred = np.array(predicted)
            gt = np.array(ground_truth)
            preds.append(pred)
            gts.append(gt)
            plt.plot(gt[:, 0], gt[:, 1], '-gv', pred[:, 0], pred[:, 1], '-rv')
            plt.plot(gt[0, 0], gt[0, 1], "gs", pred[0, 0], pred[0, 1], 'rs')
            plt.show()
        print(np.mean(np.array(ADEs)), np.mean(np.array(FDEs)))
        np.savez('ours_result.npz', pred=preds, gt=gts)
