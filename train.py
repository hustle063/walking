from utils.dataloader import NpzLoader
from utils.quaternion import normalize_quaternion, geodesic_loss, forward_kinematics
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.network import StateEncoder, LSTM, Decoder
from tqdm import tqdm
import torch
import torch.optim as optim

# TODO remove rotation 1， 6， 17， 24
if __name__ == '__main__':
    file = 'cmu_train.npz'
    npzloader = NpzLoader(file, visualize=False)
    data = DataLoader(npzloader, batch_size=32, shuffle=True)
    print('load complete')
    state_encoder = StateEncoder(in_dim=129)
    state_encoder = state_encoder.cuda()
    lstm = LSTM(in_dim=256)
    lstm = lstm.cuda()
    decoder = Decoder(in_dim=768, out_dim=129)
    decoder = decoder.cuda()
    optimizer_g = optim.Adam(lr=0.0001, params=list(state_encoder.parameters()) + list(lstm.parameters()),
                             betas=(0.5, 0.9), weight_decay=0.00001)
    writer = SummaryWriter('log')

    for epoch in range(50):
        state_encoder.train()
        lstm.train()
        decoder.train()
        rig_pred = {}
        for i_batch, sampled_batch in tqdm(enumerate(data)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            loss_q = 0
            loss_contact = 0
            loss_root = 0
            local_q = sampled_batch['local_q'].cuda()
            root_v = sampled_batch['root_v'].cuda()
            contact = sampled_batch['contact'].cuda()
            worldpos = sampled_batch['worldpos'].cuda()

            lstm.init_hidden(local_q.size(0))

            batch_size = local_q.size()[0]
            std = torch.std(worldpos, axis=(0, 1), keepdim=True)

            for t in range(239):  # window length
                if t == 0:
                    local_q_t = local_q[:, t]
                    contact_t = contact[:, t]
                    root_v_t = root_v[:, t]
                    root_pos_pred = worldpos[:, t, 0:3] + root_v_t
                elif t % 8 < 4:
                    local_q_t = local_q[:, t]
                    contact_t = contact[:, t]
                    root_v_t = root_v[:, t]
                    root_pos_pred = worldpos[:, t, 0:3] + root_v_t
                else:
                    local_q_t = local_q_pred.reshape(batch_size, -1)
                    contact_t = contact_pred[0]
                    root_v_t = root_v_pred[0]
                    root_pos_pred += root_v_t
                state_input = torch.cat([local_q_t, root_v_t, contact_t], -1).unsqueeze(0)

                h_state = state_encoder(state_input)
                h_out = lstm(h_state)
                h_pred, contact_pred = decoder(h_out)

                local_q_v_pred = h_pred[:, :, :124]
                root_v_pred = h_pred[:, :, 124:128]
                local_q_pred = local_q_t + local_q_v_pred

                local_q_next = local_q[:, t + 1].reshape(batch_size, -1, 4)

                # TODO loss function
                local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))
                loss_q += geodesic_loss(local_q_pred, local_q_next)
                root_v_next = root_v[:, t + 1]
                loss_root += torch.mean(torch.abs(root_v_next - root_v_pred[0]))
                contact_next = contact[:, t + 1]
                threshold = torch.tensor([0.5], device='cuda')
                results = (contact_pred[0] > threshold).float() * 1
                loss_contact += torch.mean(torch.abs(results - contact_next))

                rig_pred[i_batch]['q'].append(local_q_pred)
                rig_pred[i_batch]['root_pos'].append(root_pos_pred.detach().clone())

            pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                          torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                          npzloader.data['rig'], npzloader.data['edges'])

            pos_pred = pos_pred.view(pos_pred.shape[0], pos_pred.shape[1], -1)

            loss_pos = torch.mean(torch.abs(pos_pred - worldpos[:, 1:, :]) / std)

            loss_total = 1.0*loss_q + 1.0*loss_pos + 0.2*loss_root + 0.1*loss_contact
            loss_total.backward()
            optimizer_g.step()

            writer.add_scalar('loss_quat', loss_q.item(), global_step=epoch * len(data) + i_batch)
            writer.add_scalar('loss_position', loss_pos.item(), global_step=epoch * len(data) + i_batch)
            writer.add_scalar('loss_root', loss_root.item(), global_step=epoch * len(data) + i_batch)
            writer.add_scalar('loss_contact', loss_contact.item(), global_step=epoch * len(data) + i_batch)
            writer.add_scalar('loss_total', loss_total.item(), global_step=epoch * len(data) + i_batch)

        state = {
            'epoch': epoch,
            'state_encoder': state_encoder.state_dict(),
            'lstm': lstm.state_dict(),
            'decoder': decoder.state_dict(),
            'optimize': optimizer_g.state_dict()
        }
        torch.save(state, 'trained_model/checkpoint4.t7')


