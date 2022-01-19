import torch
from utils.dataloader import NpzLoader
from utils.quaternion import normalize_quaternion, forward_kinematics
from utils.visualization import render_animation
from torch.utils.data import Dataset, DataLoader
from models.network import StateEncoder, LSTM, Decoder, GRUModel, ControlEncoder
from tqdm import tqdm

if __name__ == '__main__':
    model = torch.load('E:/trained_model/checkpoint33_min_500.t7')
    file = 'edin_test_less_30fps.npz'
    npzloader = NpzLoader(file, visualize=False, offset=360)
    data = DataLoader(npzloader, batch_size=4)
    print('load complete')
    input_dim = 101

    state_encoder = StateEncoder(in_dim=input_dim)
    state_encoder = state_encoder.cuda()
    state_encoder.load_state_dict(model['state_encoder'])
    # lstm = LSTM(in_dim=256)
    # lstm = lstm.cuda()
    # lstm.load_state_dict(model['lstm'])
    control_encoder = ControlEncoder(in_dim=6, out_dim=16)
    control_encoder = control_encoder.cuda()
    control_encoder.load_state_dict(model['control_encoder'])
    gru = GRUModel(272)
    gru = gru.cuda()
    gru.load_state_dict(model['gru'])

    decoder = Decoder(in_dim=1000, out_dim=input_dim-1)
    decoder = decoder.cuda()
    decoder.load_state_dict(model['decoder'])
    rig_pred = {}

    for epoch in range(1):
        state_encoder.eval()
        # lstm.eval()
        gru.eval()
        decoder.eval()

        for i_batch, sampled_batch in tqdm(enumerate(data)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            local_q = sampled_batch['local_q'].cuda()
            root_v = sampled_batch['root_v'].cuda()
            contact = sampled_batch['contact'].cuda()
            worldpos = sampled_batch['worldpos'].cuda()
            xz_v = sampled_batch['xz_v'].cuda()
            rot_angle = sampled_batch['rot_angle'].cuda()
            rot_dir = sampled_batch['rot_dir'].cuda()

            # lstm.init_hidden(local_q.size(0))

            batch_size = local_q.size()[0]

            for t in range(59):  # window length
                if t == 0:
                    hidden = None
                    local_q_t = local_q[:, t]
                    # contact_t = contact[:, t]
                    xz_v_t = xz_v[:, t:t + 1]
                    root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                    rig_pred[i_batch]['root_pos'].append(root_pos_next)
                else:
                    local_q_t = local_q_pred.reshape(batch_size, -1)
                    # contact_t = contact_pred[0]
                    xz_v_t = xz_v_pred[0]
                    root_pos_next = worldpos[:, t, 0:3] + root_v[:, t]
                    rig_pred[i_batch]['root_pos'].append(root_pos_next)

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

                local_q_v_pred = h_pred
                local_q_pred = local_q_t + local_q_v_pred
                local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))
                rig_pred[i_batch]['q'].append(local_q_pred)

            pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                          torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                          npzloader.data['rig'], npzloader.data['edges'])

            pos_pred = pos_pred.cpu().detach().numpy()
            for i in range(pos_pred.shape[0]):
                sample = {'trajectory': pos_pred[i], 'edges': npzloader.data['edges']}
                filename = 'E:/video33/min_500/walk{}_{}.mp4'.format(i_batch, i)
                render_animation(sample, filename, fps=30)

