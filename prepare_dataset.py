# Load bvh file and convert it to the quaternion format.

import glob
from utils.bvh import BvhReader, process_bvhkeyframe, data_store


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rigs = []
    for file in glob.glob("/Users/h1y1c/Desktop/walking/dataset/edin_locomotion/*.bvh", recursive=True):
        my_bvh = BvhReader(file)
        rig = my_bvh.read()
        for i in range(rig.frames):
            new_frame = process_bvhkeyframe(rig.worldpos, rig.quaternion, rig.keyframes[i], rig.root, i)
        rig.worldpos.default_factory = None  # freeze rig.worldpos
        rig.quaternion.default_factory = None  # freeze rig.quaternion
        rigs.append(rig)
    data_store(rigs, filename='edin_train.npz')

