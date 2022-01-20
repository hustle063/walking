from models.network import StateEncoder, ControlEncoder, Decoder, GRUModel
from utils.quaternion import normalize_quaternion, forward_kinematics, geodesic_loss
from utils.visualization import render_animation
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class PhysNetwork:
    def __init__(self, input_dim=101):
        self.state_encoder = StateEncoder(in_dim=input_dim, out_dim=256)
        self.state_encoder = self.state_encoder.cuda()
        self.control_encoder = ControlEncoder(in_dim=6, out_dim=16)
        self.control_encoder = self.control_encoder.cuda()
        self.gru = GRUModel(256+16)
        self.gru = self.gru.cuda()
        self.decoder = Decoder(in_dim=1000, out_dim=input_dim-1)
        self.decoder = self.decoder.cuda()
        self.optimizer_g = optim.Adam(lr=0.0001, params=list(self.state_encoder.parameters()) +
                                      list(self.gru.parameters()) + list(self.decoder.parameters()), betas=(0.5, 0.9),
                                      weight_decay=0.0001)
        self.min_loss_pos_val_500 = np.inf
        self.min_loss_pos_val_333 = np.inf
        self.min_loss_pos_val_167 = np.inf
        self.local_q = None
        self.root_v = None
        self.contact = None
        self.worldpos = None
        self.xz_v = None
        self.rot_angle = None
        self.rot_dir = None
        self.local_q_pred = None
        self.contact_pred = None
        self.xz_v_pred = None
        self.hidden = None
        self.local_q_next = None

    def read_data(self, batch_data):
        self.local_q = batch_data['local_q'].cuda()
        self.root_v = batch_data['root_v'].cuda()
        self.contact = batch_data['contact'].cuda()
        self.worldpos = batch_data['worldpos'].cuda()
        self.xz_v = batch_data['xz_v'].cuda()
        self.rot_angle = batch_data['rot_angle'].cuda()
        self.rot_dir = batch_data['rot_dir'].cuda()
        return self.local_q.size()[0]

    def forward(self, t, batch_size, easy_state=True):
        if easy_state:
            local_q_t = self.local_q[:, t]
            contact_t = self.contact[:, t]
            xz_v_t = self.xz_v[:, t:t+1]
        else:
            local_q_t = self.local_q_pred.reshape(batch_size, -1)
            contact_t = self.contact_pred[0]
            xz_v_t = self.xz_v_pred[0]
        state_input = torch.cat([local_q_t, xz_v_t], -1).unsqueeze(0)
        control_input = torch.cat([self.rot_angle[:, t:t+1], self.rot_dir[:, t:t+1], contact_t], -1)\
            .unsqueeze(0)
        control_state = self.control_encoder(control_input)
        pred_state = self.state_encoder(state_input)
        h_state = torch.cat([pred_state, control_state], -1)
        h_out, self.hidden = self.gru(h_state, self.hidden)
        h_pred, self.contact_pred, self.xz_v_pred = self.decoder(h_out)
        local_q_v_pred = h_pred
        self.local_q_pred = local_q_t + local_q_v_pred
        self.local_q_next = self.local_q[:, t+1].reshape(batch_size, -1, 4)
        self.local_q_pred = normalize_quaternion(self.local_q_pred[0].reshape(batch_size, -1, 4))

    def train(self, npzloader_train, npzloader_val, n_epoch, batch_num, writer, window=60):
        data = DataLoader(npzloader_train, batch_size=batch_num, shuffle=True)
        for epoch in range(n_epoch):
            self.state_encoder.train()
            self.control_encoder.train()
            self.gru.train()
            self.decoder.train()
            rig_pred = {}
            for i_batch, sampled_batch in tqdm(enumerate(data)):
                rig_pred[i_batch] = {'root_pos': [], 'q': []}
                loss_q = 0
                loss_contact = 0
                loss_root = 0
                loss_pos = 0
                batch_size = self.read_data(sampled_batch)
                # std = torch.std(self.worldpos, axis=(0, 1), keepdim=True)
                self.optimizer_g.zero_grad()
                self.hidden = None
                stage = 20
                for t in range(window-1):
                    if t == 0:  # Initial start has to be in easy state
                        self.forward(t, batch_size, easy_state=True)
                    elif t % stage < stage / 2:
                        self.forward(t, batch_size, easy_state=True)
                    else:
                        self.forward(t, batch_size, easy_state=False)
                    loss_q += geodesic_loss(self.local_q_pred, self.local_q_next)
                    xz_v_next = self.xz_v[:, t+1:t+2]
                    loss_root += torch.sum(torch.mean(torch.abs(xz_v_next - self.xz_v_pred[0]), dim=0))
                    contact_next = self.contact[:, t+1:t+2]
                    # threshold = torch.tensor([0.5], device='cuda')
                    # results = (contact_pred[0] > threshold).float() * 1
                    loss_contact += torch.mean(torch.abs(self.contact_pred[0] - contact_next))
                    rig_pred[i_batch]['q'].append(self.local_q_pred)
                    rig_pred[i_batch]['root_pos'].append(self.worldpos[:, t+1, 0:3].detach().clone())

                pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                              torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                              npzloader_train.data['rig'], npzloader_train.data['edges'])

                pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)
                loss_pos = torch.mean(torch.abs(pos_pred - self.worldpos[:, 1:, :]))
                # loss_pos = torch.nn.functional.mse_loss(pos_pred, worldpos[:, 1:, :])

                loss_total = 1.0*loss_pos + 0.05*loss_q + 0.1*loss_root/60 + 0.5*loss_contact/60
                loss_total.backward()
                self.optimizer_g.step()

                writer.add_scalar('loss_quat', loss_q.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_position', loss_pos.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_root', loss_root.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_contact', loss_contact.item(), global_step=epoch * len(data) + i_batch)
                writer.add_scalar('loss_total', loss_total.item(), global_step=epoch * len(data) + i_batch)

            if epoch % 1 == 0:
                self.validation(npzloader_val, data, epoch, writer)
        self.save_model('E:/trained_model/checkpoint36_final.t7', epoch)

    def validation(self, npzloader, data_train, epoch, writer, window=20):
        notsave = True
        data_val = DataLoader(npzloader, batch_size=8, drop_last=True)
        rig_val = {}
        loss_pos_val_167 = 0
        loss_pos_val_333 = 0
        loss_pos_val_500 = 0
        self.state_encoder.eval()
        self.control_encoder.eval()
        self.gru.eval()
        self.decoder.eval()
        for j_batch, val_batch in tqdm(enumerate(data_val)):
            rig_val[j_batch] = {'root_pos': [], 'q': []}
            batch_size = self.read_data(val_batch)
            for t in range(window):  # window length
                if t == 0:
                    self.forward(t, batch_size, easy_state=True)
                else:
                    self.forward(t, batch_size, easy_state=False)
                rig_val[j_batch]['q'].append(self.local_q_pred)
                rig_val[j_batch]['root_pos'].append(self.worldpos[:, t + 1, 0:3].detach().clone())

            pos_pred = forward_kinematics(torch.stack(rig_val[j_batch]['q'], dim=1),
                                          torch.stack(rig_val[j_batch]['root_pos'], dim=1),
                                          npzloader.data['rig'], npzloader.data['edges'])

            pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)
            loss_pos_val_500 += torch.nn.functional.mse_loss(pos_pred[:, :15, :], self.worldpos[:, 1:16, :])
            loss_pos_val_333 += torch.nn.functional.mse_loss(pos_pred[:, :10, :], self.worldpos[:, 1:11, :])
            loss_pos_val_167 += torch.nn.functional.mse_loss(pos_pred[:, :5, :], self.worldpos[:, 1:6, :])

        loss_pos_val_500 = loss_pos_val_500 / (j_batch + 1)
        loss_pos_val_333 = loss_pos_val_333 / (j_batch + 1)
        loss_pos_val_167 = loss_pos_val_167 / (j_batch + 1)
        writer.add_scalar('loss_pos_val_500', loss_pos_val_500.item(), global_step=epoch * len(data_train))
        writer.add_scalar('loss_pos_val_333', loss_pos_val_333.item(), global_step=epoch * len(data_train))
        writer.add_scalar('loss_pos_val_167', loss_pos_val_167.item(), global_step=epoch * len(data_train))
        if loss_pos_val_500 < self.min_loss_pos_val_500 and notsave:
            self.save_model('E:/trained_model/checkpoint36_min_500.t7', epoch)
            print(loss_pos_val_500, epoch)
            self.min_loss_pos_val_500 = loss_pos_val_500
            notsave = False
        if loss_pos_val_333 < self.min_loss_pos_val_333 and notsave:
            self.save_model('E:/trained_model/checkpoint36_min_333.t7', epoch)
            self.min_loss_pos_val_333 = loss_pos_val_333
            notsave = False
        if loss_pos_val_167 < self.min_loss_pos_val_167 and notsave:
            self.save_model('E:/trained_model/checkpoint36_min_167.t7', epoch)
            self.min_loss_pos_val_167 = loss_pos_val_167
            notsave = False

    def save_model(self, filename, epoch):
        state = {
            'epoch': epoch,
            'state_encoder': self.state_encoder.state_dict(),
            'control_encoder': self.control_encoder.state_dict(),
            'gru': self.gru.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimize': self.optimizer_g.state_dict()
        }
        torch.save(state, filename)

    def predict(self, npzloader, model_file, batch_num, window=60):
        data = DataLoader(npzloader, batch_size=batch_num)
        model = torch.load(model_file)
        self.state_encoder.load_state_dict(model['state_encoder'])
        self.control_encoder.load_state_dict(model['control_encoder'])
        self.gru.load_state_dict(model['gru'])
        self.decoder.load_state_dict(model['decoder'])
        rig_pred = {}
        self.state_encoder.eval()
        self.control_encoder.eval()
        self.gru.eval()
        self.decoder.eval()
        for i_batch, sampled_batch in tqdm(enumerate(data)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            batch_size = self.read_data(sampled_batch)
            for t in range(window-1):  # window length
                if t == 0:
                    self.forward(t, batch_size, easy_state=True)
                else:
                    self.forward(t, batch_size, easy_state=False)
                rig_pred[i_batch]['q'].append(self.local_q_pred)
                rig_pred[i_batch]['root_pos'].append(self.worldpos[:, t + 1, 0:3].detach().clone())

            pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                          torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                          npzloader.data['rig'], npzloader.data['edges'])

            pos_pred = pos_pred.cpu().detach().numpy()
            for i in range(pos_pred.shape[0]):
                sample = {'trajectory': pos_pred[i], 'edges': npzloader.data['edges']}
                filename = 'E:/video36/min_167/walk{}_{}.mp4'.format(i_batch, i)
                render_animation(sample, filename, fps=30)










