import numpy as np
import collections


class Node(object):
    """Skeleton hierarchy node. A single vertex with potentially multiple edges."""

    def __init__(self, root=False):
        self.name = None
        self.channels = []
        self.offset = (0, 0, 0)
        self.children = []
        self._is_root = root
        self.parent = None

        # Translation matrices. The rotaion will not be included.
        self.transmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                 [0., 0., 1., 0.], [0., 0., 0., 1.]])

        self.rot = {}  # self.rot[time] Rotation values at the frame.
        self.global_transform = None  # temporarily store the global transmat

    @property
    def is_root(self):
        return self._is_root

    @property
    def is_end_site(self):
        return len(self.children) == 0


class Skeleton:
    """Represent the rig and store motion information."""

    def __init__(self, hips, keyframes=[], frames=0, frame_time=.033333333):
        self.root = hips
        # 9/1/08: we now transfer the large bvh.keyframes data structure to
        # the skeleton because we need to keep this dataset around.
        self.keyframes = keyframes
        self.frames = frames  # Number of frames (caller must set correctly)
        self.frame_time = frame_time

        # A nested dictionary of list: worldpos[frame][joint.name] = [xpos, ypos, zpos]
        self.worldpos = collections.defaultdict(dict)  # Time-based worldspace xyz position of the joints' endpoints.
        self.quaternion = collections.defaultdict(dict)
