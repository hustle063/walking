# Most code are from https://github.com/facebookresearch/QuaterNet/
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage.filters as filters
import copy

class NpzLoader(Dataset):
    def __init__(self, path, window=60, offset=-30, visualize=False):
        self.visualize = visualize
        self.real_spine_length = 0.402  # unit in meter DOI:10.1186/1756-0500-6-58
        self.x_std = 0
        self.data = self._load_npz(path, window, offset)

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

    def _get_joints_index(self, joint_names, target_name):
        try:
            idx = joint_names.index(target_name)
        except ValueError:
            print('joint {} dose not exist. Please check your bvh file'.format(target_name))
            raise ValueError
        return idx

    def _find_foot_joints(self, joints):
        lfoot_idx = self._get_joints_index(joints, 'LeftFoot')
        rfoot_idx = self._get_joints_index(joints, 'RightFoot')
        ltoe_idx = self._get_joints_index(joints, 'LeftToeBase')
        rtoe_idx = self._get_joints_index(joints, 'RightToeBase')
        return [lfoot_idx, rfoot_idx, ltoe_idx, rtoe_idx]

    def _find_facing_direction_joints(self, joints):
        lshoulder_idx = self._get_joints_index(joints, 'LeftUpLeg')
        rshoulder_idx = self._get_joints_index(joints, 'RightUpLeg')
        hip_idx = self._get_joints_index(joints, 'Neck')
        return [lshoulder_idx, rshoulder_idx, hip_idx]

    def _find_facing_direction(self, dir_idx, worldpos):
        line1 = worldpos[:, dir_idx[0], :] - worldpos[:, dir_idx[2], :]
        line2 = worldpos[:, dir_idx[1], :] - worldpos[:, dir_idx[2], :]
        forward = np.cross(line2, line1)
        direction_filterwidth = 10
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
        forward = forward[:, [0, 2]] # We only consider the direction projected on xz-plane
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        return forward

    def _find_height(self, foot_idx, worldpos):
        foot_heights = np.min(worldpos[:, foot_idx, 1], axis=1).min()
        return foot_heights

    def _extract_feet_contacts(self, pos, foot_idx, velfactor=0.02):
        [lfoot_idx, rfoot_idx, ltoe_idx, rtoe_idx] = foot_idx

        lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
        contacts_lfoot = (np.sum(lfoot_xyz, axis=-1) < velfactor)

        rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
        contacts_rfoot = (np.sum(rfoot_xyz, axis=-1) < velfactor)

        ltoe_xyz = (pos[1:, ltoe_idx, :] - pos[:-1, ltoe_idx, :]) ** 2
        contacts_ltoe = (np.sum(ltoe_xyz, axis=-1) < velfactor)

        rtoe_xyz = (pos[1:, rtoe_idx, :] - pos[:-1, rtoe_idx, :]) ** 2
        contacts_rtoe = (np.sum(rtoe_xyz, axis=-1) < velfactor)

        # Duplicate the last frame for shape consistency
        contacts_lfoot = np.concatenate([contacts_lfoot, contacts_lfoot[-1:]], axis=0)
        contacts_rfoot = np.concatenate([contacts_rfoot, contacts_rfoot[-1:]], axis=0)
        contacts_ltoe = np.concatenate([contacts_ltoe, contacts_ltoe[-1:]], axis=0)
        contacts_rtoe = np.concatenate([contacts_rtoe, contacts_rtoe[-1:]], axis=0)

        return [contacts_lfoot, contacts_ltoe, contacts_rfoot, contacts_rtoe]

    def _find_spine_length(self, pos, joints):
        hips_idx = self._get_joints_index(joints, 'Hips')
        spine_idx = self._get_joints_index(joints, 'Spine')
        spine1_idx = self._get_joints_index(joints, 'Spine1')
        neck_idx = self._get_joints_index(joints, 'Neck')
        spine_length = np.linalg.norm(pos[spine_idx, :] - pos[hips_idx, :]) + \
                       np.linalg.norm(pos[spine1_idx, :] - pos[spine_idx, :]) + \
                       np.linalg.norm(pos[neck_idx, :] - pos[spine1_idx, :])
        return spine_length

    def _find_skeleton_scale(self, pos, joints):
        skeleton_spine_length = self._find_spine_length(pos, joints)
        skeleton_scale = self.real_spine_length / skeleton_spine_length
        return skeleton_scale

    def _find_xz_velocity(self, root_velocity):
        direction_filterwidth = 10
        xz_velocity = filters.gaussian_filter1d(root_velocity[:, [0, 2]], direction_filterwidth, axis=0, mode='nearest')
        xz_scalar = np.sqrt((xz_velocity ** 2).sum(axis=-1))
        unit_xz_velocity = xz_velocity / xz_scalar[..., np.newaxis]
        return unit_xz_velocity, xz_scalar

    def _load_npz(self, path, window, offset):
        result = {}
        data = np.load(path, 'r', allow_pickle=True)
        P = []
        Q = []
        root_v = []
        contact_state = []
        actions = []
        styles = []
        rot_angle = []
        xz_v = []
        skeleton_scales = []
        root_p = []
        face_dir = []
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
            # self._find_visual_edges(rig, edges, joint_names)
            self._find_visual_edges(rig, edges, joint_names)
            skeleton_scale = self._find_skeleton_scale(worldpos[0, :, :], joint_names)
            root_velocity = self._find_root_velocity(worldpos[:, 0, :])
            y_velocity = root_velocity[:, 1]
            foot_idx = self._find_foot_joints(joint_names)
            contact = self._extract_feet_contacts(worldpos[1:, :, :], foot_idx)
            contact = np.column_stack(contact)
            dir_idx = self._find_facing_direction_joints(joint_names)
            facing_dir = self._find_facing_direction(dir_idx, worldpos[:, :, :])
            unit_xz_velocity, xz_velocity = self._find_xz_velocity(root_velocity)
            rot_angle_sin = np.cross(facing_dir, unit_xz_velocity)
            rot_angle_cos = np.multiply(facing_dir, unit_xz_velocity).sum(-1)
            end_frame = start_frame + window
            while end_frame < worldpos.shape[0]:
                # remove bias in x and z direction
                temp_P = copy.deepcopy(worldpos[start_frame:end_frame, :, :])
                root_pos = copy.deepcopy(temp_P[:, 0, :])
                temp_P[:, :, 0] = temp_P[:, :, 0] - temp_P[:, 0:1, 0]
                temp_P[:, :, 1] = temp_P[:, :, 1] - temp_P[:, 0:1, 1]
                temp_P[:, :, 2] = temp_P[:, :, 2] - temp_P[:, 0:1, 2]
                # temp_height = self._find_height(foot_idx, temp_P)
                # temp_P[:, :, 1] = temp_P[:, :, 1] - temp_height
                P.append(np.reshape(temp_P, (window, -1)))
                Q.append(np.reshape(rotations[start_frame:end_frame, :, :], (window, -1)))
                actions.append(action)
                styles.append(style)
                skeleton_scales.append(skeleton_scale)
                # root_v.append(root_velocity[start_frame:end_frame])
                root_v.append(unit_xz_velocity[start_frame:end_frame])
                root_p.append(root_pos)
                face_dir.append(facing_dir[start_frame:end_frame])
                xz_v.append(xz_velocity[start_frame:end_frame])
                rot_angle.append(np.stack((rot_angle_sin[start_frame:end_frame], rot_angle_cos[start_frame:end_frame]),
                                          axis=-1))
                contact_state.append(contact[start_frame - 1:end_frame - 1, :])
                start_frame = end_frame + offset
                end_frame = start_frame + window

        result = {
            'actions': action,
            'styles': style,
            'scales': skeleton_scales,
            'rotations': Q,
            'trajectory': P,
            'root_velocity': root_v,
            'root_position': root_p,
            'delta_rotation_angle': rot_angle,
            'xz_plane_velocity': xz_v,
            'contact_state': contact_state,
            'rig': rig,
            'edges': edges,
            'facing_direction': face_dir,
        }
        return result

    def __len__(self):
        return len(self.data['rotations'])

    def __getitem__(self, idx):
        sample = {
            'local_q': self.data['rotations'][idx].astype(np.float32),
            'root_v': self.data['root_velocity'][idx].astype(np.float32),
            'root_p': self.data['root_position'][idx].astype(np.float32),
            'contact': self.data['contact_state'][idx].astype(np.float32),
            'worldpos': self.data['trajectory'][idx].astype(np.float32),
            'rot_angle': self.data['delta_rotation_angle'][idx].astype(np.float32),
            'xz_v': self.data['xz_plane_velocity'][idx].astype(np.float32),
            'scale': self.data['scales'][idx].astype(np.float32),
            'face_dir': self.data['facing_direction'][idx].astype(np.float32)
        }
        return sample
