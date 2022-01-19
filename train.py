import numpy as np

from utils.dataloader import NpzLoader
from utils.quaternion import normalize_quaternion, geodesic_loss, forward_kinematics
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.network import StateEncoder, ControlEncoder, Decoder, GRUModel
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# TODO remove rotation 1， 6， 17， 24
if __name__ == '__main__':
    file_train = 'edin_train_less_30fps.npz'
    npzloader_train = NpzLoader(file_train, visualize=False)
    data_train = DataLoader(npzloader_train, batch_size=32, shuffle=True)
    file_val = 'edin_test_less_30fps.npz'
    npzloader_val = NpzLoader(file_val, window=21, offset=180, visualize=False)
    data_val = DataLoader(npzloader_val, batch_size=8, drop_last=True)
    print('load complete')
    # input_dim = 127 for edin_train
    # input_dim = 111 for edin_train_less
    input_dim = 101

    state_encoder = StateEncoder(in_dim=input_dim, out_dim=256)
    state_encoder = state_encoder.cuda()
    # lstm = LSTM(in_dim=256)
    # lstm = lstm.cuda()
    control_encoder = ControlEncoder(in_dim=6, out_dim=16)
    control_encoder = control_encoder.cuda()
    gru = GRUModel(272)
    gru = gru.cuda()
    decoder = Decoder(in_dim=1000, out_dim=input_dim-1)
    decoder = decoder.cuda()
    optimizer_g = optim.Adam(lr=0.0001, params=list(state_encoder.parameters()) + list(gru.parameters()) + list(decoder.parameters()),
                             betas=(0.5, 0.9), weight_decay=0.0001)
    writer = SummaryWriter('log')

    n_epoch = 120
    min_loss_pos_val_500 = np.inf
    min_loss_pos_val_333 = np.inf
    min_loss_pos_val_167 = np.inf

    for epoch in range(n_epoch):
        state_encoder.train()
        control_encoder.train()
        gru.train()
        decoder.train()
        rig_pred = {}
        for i_batch, sampled_batch in tqdm(enumerate(data_train)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            loss_q = 0
            loss_contact = 0
            loss_root = 0
            loss_pos = 0
            local_q = sampled_batch['local_q'].cuda()
            root_v = sampled_batch['root_v'].cuda()
            contact = sampled_batch['contact'].cuda()
            worldpos = sampled_batch['worldpos'].cuda()
            xz_v = sampled_batch['xz_v'].cuda()
            rot_angle = sampled_batch['rot_angle'].cuda()
            rot_dir = sampled_batch['rot_dir'].cuda()

            # lstm.init_hidden(local_q.size(0))

            batch_size = local_q.size()[0]
            std = torch.std(worldpos, axis=(0, 1), keepdim=True)

            # if epoch < 50:
            #    stage = 4
            # elif epoch < 100:
            #     stage = 4
            # elif epoch < 100:
            #     stage = 20

            stage = 20

            optimizer_g.zero_grad()
            for t in range(59):  # window length
                if t == 0:
                    hidden = None
                    local_q_t = local_q[:, t]
                    contact_t = contact[:, t]
                    xz_v_t = xz_v[:, t:t+1]
                    root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                elif t % stage < stage/2:
                    local_q_t = local_q[:, t]
                    contact_t = contact[:, t]
                    xz_v_t = xz_v[:, t:t+1]
                    root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                else:
                    local_q_t = local_q_pred.reshape(batch_size, -1)
                    contact_t = contact_pred[0]
                    xz_v_t = xz_v_pred[0]
                    root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                state_input = torch.cat([local_q_t, xz_v_t], -1).unsqueeze(0)
                #  control_input = torch.cat([contact[:, t, :], xz_v[:, t:t+1], rot_angle[:, t:t+1],
                #                            rot_dir[:, t:t+1]], -1).unsqueeze(0)
                control_input = torch.cat([rot_angle[:, t:t+1], rot_dir[:, t:t+1], contact[:, t, :]], -1).unsqueeze(0)
                control_state = control_encoder(control_input)
                pred_state = state_encoder(state_input)
                h_state = torch.cat([pred_state, control_state], -1)
                h_out, hidden = gru(h_state, hidden)
                # h_pred = decoder(h_out)
                h_pred, contact_pred, xz_v_pred = decoder(h_out)

                # local_q_v_pred = h_pred[:, :, :input_dim-3]
                # root_v_pred = h_pred[:, :, input_dim-3:input_dim+1]
                local_q_v_pred = h_pred
                local_q_pred = local_q_t + local_q_v_pred

                local_q_next = local_q[:, t + 1].reshape(batch_size, -1, 4)
                local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))

                # TODO loss function
                loss_q += geodesic_loss(local_q_pred, local_q_next)
                xz_v_next = xz_v[:, t:t+1]
                loss_root += torch.sum(torch.mean(torch.abs(xz_v_next - xz_v_pred[0]), dim=0))
                contact_next = contact[:, t + 1]
                # threshold = torch.tensor([0.5], device='cuda')
                # results = (contact_pred[0] > threshold).float() * 1
                loss_contact += torch.mean(torch.abs(contact_pred[0] - contact_next))

                rig_pred[i_batch]['q'].append(local_q_pred)
                rig_pred[i_batch]['root_pos'].append(root_pos_next.detach().clone())

            pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                          torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                          npzloader_train.data['rig'], npzloader_train.data['edges'])

            pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)
            loss_pos = torch.mean(torch.abs(pos_pred - worldpos[:, 1:, :]))
            # loss_pos = torch.nn.functional.mse_loss(pos_pred, worldpos[:, 1:, :])

            loss_total = 1.0*loss_pos + 0.1*loss_root/60 + 0.5*loss_contact/60
            loss_total.backward()
            optimizer_g.step()

            writer.add_scalar('loss_quat', loss_q.item(), global_step=epoch * len(data_train) + i_batch)
            writer.add_scalar('loss_position', loss_pos.item(), global_step=epoch * len(data_train) + i_batch)
            writer.add_scalar('loss_root', loss_root.item(), global_step=epoch * len(data_train) + i_batch)
            writer.add_scalar('loss_contact', loss_contact.item(), global_step=epoch * len(data_train) + i_batch)
            writer.add_scalar('loss_total', loss_total.item(), global_step=epoch * len(data_train) + i_batch)

        # validation:
        if epoch % 1 == 0:
            NOTSAVE = True
            rig_val = {}
            loss_pos_val_167 = 0
            loss_pos_val_333 = 0
            loss_pos_val_500 = 0
            state_encoder.eval()
            control_encoder.eval()
            gru.eval()
            decoder.eval()
            for j_batch, val_batch in tqdm(enumerate(data_val)):
                rig_val[j_batch] = {'root_pos': [], 'q': []}
                local_q = val_batch['local_q'].cuda()
                root_v = val_batch['root_v'].cuda()
                contact = val_batch['contact'].cuda()
                worldpos = val_batch['worldpos'].cuda()
                xz_v = val_batch['xz_v'].cuda()
                rot_angle = val_batch['rot_angle'].cuda()
                rot_dir = val_batch['rot_dir'].cuda()

                batch_size = local_q.size()[0]
                for t in range(20):  # window length
                    if t == 0:
                        hidden = None
                        local_q_t = local_q[:, t]
                        contact_t = contact[:, t]
                        xz_v_t = xz_v[:, t:t+1]
                        root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                    else:
                        local_q_t = local_q_pred.reshape(batch_size, -1)
                        contact_t = contact_pred[0]
                        xz_v_t = xz_v_pred[0]
                        root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                    state_input = torch.cat([local_q_t, xz_v_t], -1).unsqueeze(0)
                    # control_input = torch.cat([contact[:, t, :], xz_v[:, t:t + 1], rot_angle[:, t:t + 1],
                    #                            rot_dir[:, t:t + 1]], -1).unsqueeze(0)

                    # control_state = control_encoder(control_input)
                    control_input = torch.cat([rot_angle[:, t:t + 1], rot_dir[:, t:t + 1], contact[:, t, :]], -1).unsqueeze(0)
                    control_state = control_encoder(control_input)
                    pred_state = state_encoder(state_input)
                    h_state = torch.cat([pred_state, control_state], -1)
                    h_out, hidden = gru(h_state, hidden)

                    # h_pred, contact_pred, root_v_pred = decoder(h_out)
                    h_pred, contact_pred, xz_v_pred = decoder(h_out)

                    # h_pred, contact_pred = decoder(h_out)
                    local_q_v_pred = h_pred
                    # local_q_v_pred = h_pred[:, :, :input_dim-3]
                    # root_v_pred = h_pred[:, :, input_dim-3:input_dim+1]
                    local_q_pred = local_q_t + local_q_v_pred

                    local_q_next = local_q[:, t + 1].reshape(batch_size, -1, 4)
                    local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))
                    rig_val[j_batch]['q'].append(local_q_pred)
                    rig_val[j_batch]['root_pos'].append(root_pos_next.detach().clone())

                pos_pred = forward_kinematics(torch.stack(rig_val[j_batch]['q'], dim=1),
                                              torch.stack(rig_val[j_batch]['root_pos'], dim=1),
                                              npzloader_val.data['rig'], npzloader_val.data['edges'])

                pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)
                loss_pos_val_500 += torch.nn.functional.mse_loss(pos_pred[:, :15, :], worldpos[:, 1:16, :])
                loss_pos_val_333 += torch.nn.functional.mse_loss(pos_pred[:, :10, :], worldpos[:, 1:11, :])
                loss_pos_val_167 += torch.nn.functional.mse_loss(pos_pred[:, :5, :], worldpos[:, 1:6, :])

            loss_pos_val_500 = loss_pos_val_500 / (j_batch+1)
            loss_pos_val_333 = loss_pos_val_333 / (j_batch+1)
            loss_pos_val_167 = loss_pos_val_167 / (j_batch+1)
            writer.add_scalar('loss_pos_val_500', loss_pos_val_500.item(), global_step=epoch * len(data_train))
            writer.add_scalar('loss_pos_val_333', loss_pos_val_333.item(), global_step=epoch * len(data_train))
            writer.add_scalar('loss_pos_val_167', loss_pos_val_167.item(), global_step=epoch * len(data_train))

            if loss_pos_val_500 < min_loss_pos_val_500 and NOTSAVE:
                state = {
                    'epoch': epoch,
                    'state_encoder': state_encoder.state_dict(),
                    'control_encoder': control_encoder.state_dict(),
                    'gru': gru.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimize': optimizer_g.state_dict()
                }
                torch.save(state, 'E:/trained_model/checkpoint33_min_500.t7')
                print(loss_pos_val_500, epoch)
                min_loss_pos_val_500 = loss_pos_val_500
                NOTSAVE = False

            if loss_pos_val_333 < min_loss_pos_val_333 and NOTSAVE:
                state = {
                    'epoch': epoch,
                    'state_encoder': state_encoder.state_dict(),
                    'control_encoder': control_encoder.state_dict(),
                    'gru': gru.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimize': optimizer_g.state_dict()
                }
                torch.save(state, 'E:/trained_model/checkpoint33_min_333.t7')
                min_loss_pos_val_333 = loss_pos_val_333
                NOTSAVE = False

            if loss_pos_val_167 < min_loss_pos_val_167 and NOTSAVE:
                state = {
                    'epoch': epoch,
                    'state_encoder': state_encoder.state_dict(),
                    'control_encoder': control_encoder.state_dict(),
                    'gru': gru.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimize': optimizer_g.state_dict()
                }
                torch.save(state, 'E:/trained_model/checkpoint33_min_167.t7')
                min_loss_pos_val_167 = loss_pos_val_167
                NOTSAVE = False

    state = {
        'epoch': epoch,
        'state_encoder': state_encoder.state_dict(),
        'control_encoder': control_encoder.state_dict(),
        'gru': gru.state_dict(),
        'decoder': decoder.state_dict(),
        'optimize': optimizer_g.state_dict()
    }
    torch.save(state, 'E:/trained_model/checkpoint33_final.t7')


