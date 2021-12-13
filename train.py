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
    file_train = 'edin_train_less.npz'
    npzloader_train = NpzLoader(file_train, visualize=False)
    data_train = DataLoader(npzloader_train, batch_size=32, shuffle=True)
    file_val = 'edin_test_less.npz'
    npzloader_val = NpzLoader(file_val, window=61, offset=540, visualize=False)
    data_val = DataLoader(npzloader_val, batch_size=32)
    print('load complete')

    state_encoder = StateEncoder(in_dim=111)
    state_encoder = state_encoder.cuda()
    # lstm = LSTM(in_dim=256)
    # lstm = lstm.cuda()
    control_encoder = ControlEncoder()
    control_encoder = control_encoder.cuda()
    gru = GRUModel(320)
    gru = gru.cuda()
    decoder = Decoder(in_dim=1000, out_dim=111)
    decoder = decoder.cuda()
    optimizer_g = optim.Adam(lr=0.0001, params=list(state_encoder.parameters()) + list(gru.parameters()) + list(decoder.parameters()),
                             betas=(0.5, 0.9), weight_decay=0.0001)
    writer = SummaryWriter('log')

    n_epoch = 250

    for epoch in range(n_epoch):
        state_encoder.train()
        control_encoder.train()
        gru.train()
        decoder.train()
        rig_pred = {}
        for i_batch, sampled_batch in tqdm(enumerate(data_train)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            loss_q = 0
            # loss_contact = 0
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
            for t in range(119):  # window length
                if t == 0:
                    hidden = None
                    local_q_t = local_q[:, t]
                    # contact_t = contact[:, t]
                    root_v_t = root_v[:, t]
                    root_pos_pred = worldpos[:, t, 0:3] + root_v_t
                elif t % stage < stage/2:
                    local_q_t = local_q[:, t]
                    # contact_t = contact[:, t]
                    root_v_t = root_v[:, t]
                    root_pos_pred = worldpos[:, t, 0:3] + root_v_t
                else:
                    local_q_t = local_q_pred.reshape(batch_size, -1)
                    # contact_t = contact_pred[0]
                    root_v_t = root_v_pred[0]
                    root_pos_pred += root_v_t
                state_input = torch.cat([local_q_t, root_v_t], -1).unsqueeze(0)
                control_input = torch.cat([contact[:, t, :], xz_v[:, t:t+1], rot_angle[:, t:t+1],
                                           rot_dir[:, t:t+1]], -1).unsqueeze(0)

                control_state = control_encoder(control_input)
                pred_state = state_encoder(state_input)
                h_state = torch.cat([pred_state, control_state], -1)
                h_out, hidden = gru(h_state, hidden)
                h_pred = decoder(h_out)
                # h_pred, contact_pred = decoder(h_out)

                local_q_v_pred = h_pred[:, :, :108]
                root_v_pred = h_pred[:, :, 108:112]
                local_q_pred = local_q_t + local_q_v_pred

                local_q_next = local_q[:, t + 1].reshape(batch_size, -1, 4)
                local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))

                # TODO loss function
                loss_q += geodesic_loss(local_q_pred, local_q_next)
                root_v_next = root_v[:, t + 1]
                loss_root += torch.sum(torch.mean(torch.abs(root_v_next - root_v_pred[0]), dim=0))
                # contact_next = contact[:, t + 1]
                # threshold = torch.tensor([0.5], device='cuda')
                # results = (contact_pred[0] > threshold).float() * 1
                # loss_contact += torch.mean(torch.abs(contact_pred[0] - contact_next))

                rig_pred[i_batch]['q'].append(local_q_pred)
                rig_pred[i_batch]['root_pos'].append(root_pos_pred.detach().clone())

            pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                          torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                          npzloader_train.data['rig'], npzloader_train.data['edges'])

            pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)
            loss_pos = torch.mean(torch.abs(pos_pred - worldpos[:, 1:, :]))
            # loss_pos = torch.nn.functional.mse_loss(pos_pred, worldpos[:, 1:, :])

            loss_total = 1.0*loss_q + 1.0*loss_pos + loss_root/240
            loss_total.backward()
            optimizer_g.step()

            writer.add_scalar('loss_quat', loss_q.item(), global_step=epoch * len(data_train) + i_batch)
            writer.add_scalar('loss_position', loss_pos.item(), global_step=epoch * len(data_train) + i_batch)
            writer.add_scalar('loss_root', loss_root.item(), global_step=epoch * len(data_train) + i_batch)
            # writer.add_scalar('loss_contact', loss_contact.item(), global_step=epoch * len(data) + i_batch)
            writer.add_scalar('loss_total', loss_total.item(), global_step=epoch * len(data_train) + i_batch)

        # validation:
        if epoch % 5 == 0:
            rig_val = {}
            loss_pos_val_20 = 0
            loss_pos_val_40 = 0
            loss_pos_val_60 = 0
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
                for t in range(60):  # window length
                    if t == 0:
                        hidden = None
                        local_q_t = local_q[:, t]
                        # contact_t = contact[:, t]
                        root_v_t = root_v[:, t]
                        root_pos_pred = worldpos[:, t, 0:3] + root_v_t
                    else:
                        local_q_t = local_q_pred.reshape(batch_size, -1)
                        # contact_t = contact_pred[0]
                        root_v_t = root_v_pred[0]
                        root_pos_pred += root_v_t
                    state_input = torch.cat([local_q_t, root_v_t], -1).unsqueeze(0)
                    control_input = torch.cat([contact[:, t, :], xz_v[:, t:t + 1], rot_angle[:, t:t + 1],
                                               rot_dir[:, t:t + 1]], -1).unsqueeze(0)

                    control_state = control_encoder(control_input)
                    pred_state = state_encoder(state_input)
                    h_state = torch.cat([pred_state, control_state], -1)

                    h_out, hidden = gru(h_state, hidden)
                    h_pred = decoder(h_out)
                    # h_pred, contact_pred = decoder(h_out)

                    local_q_v_pred = h_pred[:, :, :108]
                    root_v_pred = h_pred[:, :, 108:112]
                    local_q_pred = local_q_t + local_q_v_pred

                    local_q_next = local_q[:, t + 1].reshape(batch_size, -1, 4)
                    local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))
                    rig_val[j_batch]['q'].append(local_q_pred)
                    rig_val[j_batch]['root_pos'].append(root_pos_pred.detach().clone())

                pos_pred = forward_kinematics(torch.stack(rig_val[j_batch]['q'], dim=1),
                                              torch.stack(rig_val[j_batch]['root_pos'], dim=1),
                                              npzloader_val.data['rig'], npzloader_val.data['edges'])

                pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)
                loss_pos_val_60 += torch.nn.functional.mse_loss(pos_pred[:, :60, :], worldpos[:, 1:61, :])
                loss_pos_val_40 += torch.nn.functional.mse_loss(pos_pred[:, :40, :], worldpos[:, 1:41, :])
                loss_pos_val_20 += torch.nn.functional.mse_loss(pos_pred[:, :20, :], worldpos[:, 1:21, :])

            loss_pos_val_60 = loss_pos_val_60 / j_batch
            loss_pos_val_40 = loss_pos_val_40 / j_batch
            loss_pos_val_20 = loss_pos_val_20 / j_batch
            writer.add_scalar('loss_pos_val_60', loss_pos_val_60.item(), global_step=epoch * len(data_train))
            writer.add_scalar('loss_pos_val_40', loss_pos_val_40.item(), global_step=epoch * len(data_train))
            writer.add_scalar('loss_pos_val_20', loss_pos_val_20.item(), global_step=epoch * len(data_train))


        state = {
            'epoch': epoch,
            'state_encoder': state_encoder.state_dict(),
            'gru': gru.state_dict(),
            'decoder': decoder.state_dict(),
            'optimize': optimizer_g.state_dict()
        }
        torch.save(state, 'trained_model/checkpoint15.t7')


