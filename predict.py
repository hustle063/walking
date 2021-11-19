import torch
from utils.dataloader import NpzLoader
from utils.quaternion import normalize_quaternion, forward_kinematics
from utils.visualization import render_animation
from torch.utils.data import Dataset, DataLoader
from models.network import StateEncoder, LSTM, Decoder
from tqdm import tqdm

if __name__ == '__main__':
    model = torch.load('trained_model/checkpoint4.t7')
    file = 'cmu_test.npz'
    npzloader = NpzLoader(file, visualize=False)
    data = DataLoader(npzloader, batch_size=2)
    print('load complete')
    state_encoder = StateEncoder(in_dim=129)
    state_encoder = state_encoder.cuda()
    state_encoder.load_state_dict(model['state_encoder'])
    lstm = LSTM(in_dim=256)
    lstm = lstm.cuda()
    lstm.load_state_dict(model['lstm'])
    decoder = Decoder(in_dim=768, out_dim=129)
    decoder = decoder.cuda()
    decoder.load_state_dict(model['decoder'])
    rig_pred = {}

    for epoch in range(1):
        state_encoder.eval()
        lstm.eval()
        decoder.eval()

        for i_batch, sampled_batch in tqdm(enumerate(data)):
            rig_pred[i_batch] = {'root_pos': [], 'q': []}
            local_q = sampled_batch['local_q'].cuda()
            root_v = sampled_batch['root_v'].cuda()
            contact = sampled_batch['contact'].cuda()
            worldpos = sampled_batch['worldpos'].cuda()

            lstm.init_hidden(local_q.size(0))

            batch_size = local_q.size()[0]

            for t in range(239):  # window length
                if t == 0:
                    local_q_t = local_q[:, t]
                    contact_t = contact[:, t]
                    root_v_t = root_v[:, t]
                    root_pos_pred = worldpos[:, t, 0:3] + root_v_t
                    rig_pred[i_batch]['q'].append(local_q[:, t].reshape(batch_size, -1, 4))
                    rig_pred[i_batch]['root_pos'].append(root_pos_pred)
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
                local_q_pred = normalize_quaternion(local_q_pred[0].reshape(batch_size, -1, 4))
                rig_pred[i_batch]['q'].append(local_q_pred)
                rig_pred[i_batch]['root_pos'].append(root_pos_pred.detach().clone())

            pos_pred = forward_kinematics(torch.stack(rig_pred[i_batch]['q'], dim=1),
                                          torch.stack(rig_pred[i_batch]['root_pos'], dim=1),
                                          npzloader.data['rig'], npzloader.data['edges'])

            pos_pred = pos_pred.cpu().detach().numpy()
            for i in range(pos_pred.shape[0]):
                sample = {'trajectory': pos_pred[i], 'edges': npzloader.data['edges']}
                filename = 'video/walk{}_{}.mp4'.format(i_batch, i)
                render_animation(sample, filename)

