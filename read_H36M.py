import copy
import numpy as np
from utils import H36M_utils
from utils.quaternion import qfix
from utils.bvh import BvhReader


def read_all_data(actions, data_dir, one_hot):
    """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
  """

    train_subject_ids = [1, 6, 7, 8, 9, 11]
    test_subject_ids = [5]

    train_set, complete_train = H36M_utils.load_data(data_dir, train_subject_ids, actions, one_hot)
    test_set, complete_test = H36M_utils.load_data(data_dir, test_subject_ids, actions, one_hot)

    return train_set, test_set


def revert_coordinate_space(channels, R0, T0):
    """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for ii in range(n):
        R_diff = H36M_utils.expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[ii, rootRotInd] = H36M_utils.rotmat2expmap(R)
        T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii, :3], [3, 1]))).reshape(-1)
        channels_rec[ii, :3] = T
        T_prev = T
        R_prev = R

    return channels_rec


def fkl(angles, parent, offset, rotInd, expmapInd):
    """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    nlines = 25
    qStruct = [dict() for x in range(nlines)]
    xyzStruct = [dict() for x in range(njoints)]

    counter = 0

    for i in np.arange(njoints):

        if not rotInd[i]:  # If the list is empty
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rotInd[i][0] - 1]
            yangle = angles[rotInd[i][1] - 1]
            zangle = angles[rotInd[i][2] - 1]

        r = angles[expmapInd[i]]

        thisRotation = H36M_utils.expmap2rotmat(r)
        if rotInd[i]:
            thisQuaternion = H36M_utils.rotmat2quat(thisRotation)
            qStruct[counter] = thisQuaternion
            counter += 1

        # Fix the position error according to https://github.com/una-dinosauria/human-motion-prediction/issues/23
        thisPosition = np.array([angles[0], angles[1], angles[2]])

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = offset[i, :].dot(xyzStruct[parent[i]]['rotation']) + xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    # xyz = xyz[:, [0, 2, 1]]
    quaternion = [qStruct[i] for i in range(counter)]
    quaternion = np.array(quaternion).squeeze()
    xyz = xyz[:, [2, 1, 0]]
    return xyz, quaternion


def _some_variables():
    """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def _save_npz(data_set, root, filename):
    parent, offset, rotInd, expmapInd = _some_variables()
    xyz = {}
    q = {}
    locomotion_skeletons = []
    locomotion_styles = []
    locomotion_actions = []
    locomotion_pos = []
    locomotion_rot = []
    for i in data_set:
        nframes = data_set[i].shape[0]
        xyz[i], q[i] = np.zeros((nframes, 32, 3)), np.zeros((nframes, 25, 4))
        data_set[i] = revert_coordinate_space(data_set[i], np.eye(3), np.zeros(3))
        for j in range(nframes):
            # TODO first three element [0, 1, 2] should be swapped as [1, 2, 0]
            xyz[i][j, :], q[i][j, :, :] = fkl(data_set[i][j, :], parent, offset, rotInd, expmapInd)
        q[i] = qfix(q[i])
        q[i][:, :, [0, 3, 2, 1]] = q[i][:, :, [0, 1, 2, 3]]
        locomotion_skeletons.append(root)
        locomotion_styles.append('unknown')
        locomotion_actions.append('walking')
        locomotion_pos.append(xyz[i])
        locomotion_rot.append(q[i])
    np.savez_compressed(filename,
                        worldpos=locomotion_pos,
                        rotations=locomotion_rot,
                        styles=locomotion_styles,
                        actions=locomotion_actions,
                        skeletons=locomotion_skeletons)


file = "./dataset/h3.6m/h36m_skeleton.bvh"
my_bvh = BvhReader(file)
with open(file, 'r') as my_bvh._file_handle:
  my_bvh.read_hierarchy()
  root = my_bvh.root

data_dir = './dataset/h3.6m/expmap'
train_set, test_set = read_all_data(["walking"], data_dir, False)
# _save_npz(train_set, root, 'h36m_train.npz')
_save_npz(test_set, root, 'h36m_test.npz')

print('success')
