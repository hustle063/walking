# Most code are from https://github.com/facebookresearch/QuaterNet/
import numpy
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NpzLoader(Dataset):
    def __init__(self, path, visualize=False):
        self.visualize = visualize
        self.x_std = 0
        self.data = self._load_npz(path)

    def _find_edges(self, joint, edges, joint_names):
        for child in joint.children:
            if child.children:
                edges.append((joint_names.index(joint.name), joint_names.index(child.name)))
                self._find_edges(child, edges, joint_names)

    def _find_visual_edges(self, joint, edges, joint_names):
        for child in joint.children:
            edges.append((joint_names.index(joint.name), joint_names.index(child.name)))
            self._find_visual_edges(child, edges, joint_names)

    def _find_joint_name(self, joint, joint_names):
        joint_names.append(joint.name)
        for child in joint.children:
            self._find_joint_name(child, joint_names)
            
    def _find_root_velocity(self, rootpos):
        temppos = rootpos[:-1]
        temppos = np.insert(temppos, 0, rootpos[0], 0)
        root_velocity = rootpos - temppos
        # root_velocity = np.linalg.norm(temppos, axis=1)
        return root_velocity

    def _find_foot_joints(self, joints):
        try:
            lfoot_idx = joints.index('LeftFoot')
        except ValueError:
            print('joint LeftFoot dose not exist. Please check your bvh file')
            raise ValueError
        try:
            rfoot_idx = joints.index('RightFoot')
        except ValueError:
            print('joint RightFoot dose not exist. Please check your bvh file')
            raise ValueError
        return lfoot_idx, rfoot_idx

    def _extract_feet_contacts(self, pos, lfoot_idx, rfoot_idx, velfactor=0.02):
        """
        Extracts binary tensors of feet contacts
        :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
        :param lfoot_idx: indices list of left foot joints
        :param rfoot_idx: indices list of right foot joints
        :param velfactor: velocity threshold to consider a joint moving or not
        :return: binary tensors of left foot contacts and right foot contacts
        """
        lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
        contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

        rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
        contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

        # Duplicate the last frame for shape consistency
        contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
        contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

        return contacts_l, contacts_r

    def _load_npz(self, path, window=240, offset=-120):
        result = {}
        data = np.load(path, 'r', allow_pickle=True)
        P = []
        Q = []
        root_v = []
        contact_state = []
        actions = []
        styles = []
        for (worldpos, rotations, style, action, rig) in zip(data['worldpos'], data['rotations'], data['styles'],
                                                             data['actions'], data['skeletons']):
            start_frame = 1  # We exclude the first frame which is a reference frame in CMU dataset
            joint_names = []
            self._find_joint_name(rig, joint_names)
            edges = [(-1, joint_names.index(rig.name))]
            # if self.visualize:
            #     self._find_visual_edges(rig, edges, joint_names)
            # else:
            #     self._find_edges(rig, edges, joint_names)
            self._find_visual_edges(rig, edges, joint_names)
            root_velocity = self._find_root_velocity(worldpos[1:, 0, :])
            lfoot_idx, rfoot_idx = self._find_foot_joints(joint_names)
            contact_l, contact_r = self._extract_feet_contacts(worldpos[1:, :, :], lfoot_idx, rfoot_idx)
            contact = np.column_stack((contact_l, contact_r))
            end_frame = start_frame + window
            while end_frame < worldpos.shape[0]:
                P.append(np.reshape(worldpos[start_frame:end_frame, :, :], (window, -1)))
                Q.append(np.reshape(rotations[start_frame:end_frame, :, :], (window, -1)))
                actions.append(action)
                styles.append(style)
                root_v.append(root_velocity[start_frame:end_frame])
                contact_state.append(contact[start_frame-1:end_frame-1, :])
                start_frame = end_frame + offset
                end_frame = start_frame + window

            
        result = {
            'actions': action,
            'styles': style,
            'rotations': Q,
            'trajectory': P,
            'root_velocity': root_v,
            'contact_state': contact_state,
            'rig': rig,
            'edges': edges,
            }
        return result

    def __len__(self):
        return len(self.data['rotations'])

    def __getitem__(self, idx):
        sample = {
            'local_q': self.data['rotations'][idx].astype(np.float32),
            'root_v': self.data['root_velocity'][idx].astype(np.float32),
            'contact': self.data['contact_state'][idx].astype(np.float32),
            'worldpos': self.data['trajectory'][idx].astype(np.float32)
        }
        return sample



