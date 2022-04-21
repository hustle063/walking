# Load bvh file and convert it to the quaternion format.

import glob
from utils.bvh import BvhReader, process_bvhkeyframe, data_store, remove_joints


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rigs = []
    names = []
    remove_list = ['RightHandIndex1', 'RThumb', 'LeftHandIndex1', 'LThumb',
                   'RightHandIndex1End', 'RThumbEnd', 'LeftHandIndex1End', 'LThumbEnd']
    for file in glob.glob("/Users/h1y1c/walking/dataset/h3.6m/predict_result/*.bvh", recursive=True):
        my_bvh = BvhReader(file)
        rig = my_bvh.read()
        for i in range(0, rig.frames, 2):
            new_frame = process_bvhkeyframe(rig.worldpos, rig.quaternion, rig.keyframes[i], rig.root, i)
        remove_joints(rig, remove_list)
        rig.worldpos.default_factory = None  # freeze rig.worldpos
        rig.quaternion.default_factory = None  # freeze rig.quaternion

        # def get_name(node, name_list):
        #     for child1 in node.children:
        #         name_list.append(child1.name)
        #         get_name(child1, name_list)
        #
        # for j in rigs:
        #     name = [j.root.name]
        #     get_name(j.root, name)
        #     names.append(name)

        rigs.append(rig)
    data_store(rigs, filename='h3.6m_predict.npz')


