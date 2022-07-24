from models.network import PaceEncoder, ControlEncoder, PaceDecoder, GRUModel1, PaceBlock
from utils.quaternion import normalize_quaternion, forward_kinematics, geodesic_loss, qeuler, euler_to_quaternion
from utils.visualization import render_animation
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
import math
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
                                      list(self.rightleg_layer.parameters()) + list(self.leftarm_layer.parameters()) +
                                      list(self.rightarm_layer.parameters()) + list(self.torso_layer.parameters()) +
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
        self.rot_angle_norm = None
        self.local_q_next = None
        self.skeleton_scale = None
        self.h_leftleg = None
        self.h_rightleg = None
        self.h_torso = None
        self.h_leftarm = None
        self.h_rightarm = None

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
        self.rot_angle_pred = self.decoder(h_out)

    def train(self, npzloader_train, npzloader_val, n_epoch, batch_num, writer, window=50):
        loss = torch.nn.L1Loss(reduction='sum')
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
                # std = torch.std(self.worldpos, axis=(0, 1), keepdim=True)
                batch_size = self.read_data(sampled_batch)
                self.optimizer_g.zero_grad()
                self.h_leftleg = None
                self.h_rightleg = None
                self.h_torso = None
                self.h_leftarm = None
                self.h_rightarm = None
                stage = None
                for t in range(window-1):
                    seed = np.random.uniform(0, 1)
                    if seed > (math.log10(epoch+1)/6 + 0.3):
                        stage = True
                    else:
                        stage = False
                    if t == 0:  # Initial start has to be in easy state
                        self.forward(t, easy_state=True)
                    elif stage:
                        self.forward(t, easy_state=True)
                    else:
                        self.forward(t, easy_state=False)
                    sin_angle = self.rot_angle[:, t+1, 0]
                    cos_angle = self.rot_angle[:, t+1, 1]
                    # new_sin_angle = math.sqrt(2) / 2 * (sin_angle+cos_angle)
                    # new_cos_angle = math.sqrt(2) / 2 * (cos_angle-sin_angle)

                    xz_v_next = self.xz_v[:, t+1:t+2]
                    sin_angle_pred = self.rot_angle_pred[0, :, 0]
                    cos_angle_pred = self.rot_angle_pred[0, :, 1]

                    if epoch % 100 == 0:
                        if t == 0:
                            with open('value1.txt', 'a') as file_object:
                                file_object.write('epoch {}:\n'.format(epoch))
                        # print(self.xz_v_pred[0].clone().detach().cpu().numpy(), xz_v_next)
                        with open('value1.txt', 'a') as file_object:
                            file_object.write('sin angle timestep {} stage {}\n'.format(t, stage))
                        with open('value1.txt', 'ab') as file_object:
                            sin = np.stack((self.rot_angle_pred[0, :, 0].clone().detach().cpu().numpy(),
                                            sin_angle.clone().cpu().numpy()))
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

                    loss_rot += torch.mean(xz_v_next * (loss(sin_angle_pred, sin_angle) +
                                                        loss(cos_angle_pred, cos_angle)))

                loss_rot = loss_rot/(window-1)
                loss_total = loss_rot
                loss_total.backward(retain_graph=True)
                self.optimizer_g.step()

                # writer.add_scalar('loss_xz', loss_xz.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_rotation', loss_rot.item(), global_step=epoch * len(data) + i_batch)
                # writer.add_scalar('loss_root', loss_root.item(), global_step=epoch * len(data) + i_batch)
                # writer.add_scalar('loss_total', loss_total.item(), global_step=epoch * len(data) + i_batch)

            if epoch % 10 == 0:
                # print(self.xz_v_pred[0].clone().detach().cpu().numpy(), xz_v_next)
                print(self.rot_angle_pred[0, :, 1].clone().detach().cpu().numpy(), cos_angle)
                print(self.rot_angle_pred[0, :, 0].clone().detach().cpu().numpy(), sin_angle)
            if epoch % 1 == 0:
                self.validation(npzloader_val, data, epoch, writer)
        self.save_model('E:/trained_model/h36m_pace_10_final.t7', epoch)

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
            traj_point = self.root_p[:, 0, [0, 2]]
            loss_rot = 0
            self.hidden = None
            for t in range(window):  # window length
                if t == 0:
                    self.forward(t, easy_state=True)
                else:
                    self.forward(t, easy_state=False)
                sin_angle = self.rot_angle[:, t+1, 0]
                cos_angle = self.rot_angle[:, t+1, 1]

                xz_v_next = self.xz_v[:, t+1:t+2]
                sin_angle_pred = self.rot_angle_pred[0, :, 0]
                cos_angle_pred = self.rot_angle_pred[0, :, 1]
                # loss_xz += torch.mean(torch.abs(xz_v_next - self.xz_v_pred[0]).squeeze(-1))
                loss_rot += torch.mean(xz_v_next * (
                            torch.abs(sin_angle - sin_angle_pred) + torch.abs(cos_angle - cos_angle_pred)))

                # print(sin_angle, cos_angle)

            # loss_xz = loss_xz/(window-1)
            loss_rot = loss_rot/(window-1)

        writer.add_scalar('loss_rot_val', loss_rot.item(), global_step=epoch * len(data_train))

    def save_model(self, filename, epoch):
        state = {
            'epoch': epoch,
            # 'state_encoder': self.state_encoder.state_dict(),
            # 'control_encoder': self.control_encoder.state_dict(),
            'leftleg_layer': self.leftleg_layer,
            'rightleg_layer': self.rightleg_layer,
            'torso_layer': self.torso_layer,
            'leftarm_layer': self.leftarm_layer,
            'rightarm_layer': self.rightarm_layer,
            'decoder': self.decoder.state_dict(),
            'optimize': self.optimizer_g.state_dict()
        }
        torch.save(state, filename)

    def predict(self, npzloader, model_file, batch_num, window=50):
        np.set_printoptions(suppress=True)
        data = DataLoader(npzloader, batch_size=batch_num, shuffle=False)
        model = torch.load(model_file)
        # self.state_encoder.load_state_dict(model['state_encoder'])
        # self.control_encoder.load_state_dict(model['control_encoder'])
        self.gru.load_state_dict(model['gru'])
        self.decoder.load_state_dict(model['decoder'])
        rig_pred = {}
        errors = []
        # self.state_encoder.eval()
        # self.control_encoder.eval()
        self.leftleg_layer.eval()
        self.rightleg_layer.eval()
        self.torso_layer.eval()
        self.leftarm_layer.eval()
        self.rightarm_layer.eval()
        self.decoder.eval()
        for i_batch, sampled_batch in tqdm(enumerate(data)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            batch_size = self.read_data(sampled_batch)
            traj_point = self.root_p[:, 0, [0, 2]]
            euler_pred_seq = np.zeros((batch_size, window-1, 78))
            self.hidden = None
            ground_truth = []
            predicted = []

            for t in range(window-1):  # window length
                if t < 5:
                    self.forward(t, easy_state=True)
                else:
                    self.forward(t, easy_state=False)
                xz_v_next = self.xz_v[:, t+1:t+2]
                sin_angle_pred = self.rot_angle_pred[0, :, 0]
                cos_angle_pred = self.rot_angle_pred[0, :, 1]
                face_dir = torch.clone(self.face_dir[:, t+1, :]).unsqueeze(-1)
                rot_pred = torch.stack([torch.stack([cos_angle_pred, -sin_angle_pred]),
                                   torch.stack([sin_angle_pred, cos_angle_pred])])
                xz_dir = torch.bmm(rot_pred.T, face_dir).squeeze(-1)
                xz_v_next = self.xz_v[:, t + 1:t + 2].squeeze(-1)/self.skeleton_scale
                xz_v_pred = xz_v_next.view(-1, 1).expand_as(xz_dir)
                distance = xz_v_pred * xz_dir
                traj_point = distance + traj_point
                predicted.append(traj_point[4, :].clone().detach().cpu().numpy())
                ground_truth.append(self.root_p[4, t + 1, [0, 2]].clone().cpu().numpy())

                # error = torch.mean(torch.sqrt(((traj_point - self.root_p[:, t + 1, [0, 2]]) ** 2).sum(-1)))
                # mean_errors = error.cpu().detach().numpy()
                # print(cos_angle_pred.clone().detach().cpu().numpy(), self.rot_angle[:, t + 1, 1])
                # print(sin_angle_pred.clone().detach().cpu().numpy(), self.rot_angle[:, t + 1, 0])
                # errors.append(mean_errors)
            pred = np.array(predicted)
            gt = np.array(ground_truth)
            plt.plot(pred[:25, 0], pred[:25, 1], gt[:25, 0], gt[:25, 1])
            plt.show()




