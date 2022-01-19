# Most codes are from https://github.com/tekulvw/bvh-converter

import numpy as np
import utils.quaternion as q
from math import radians, cos, sin
from utils.skeleton import Skeleton, Node


class BvhReader(object):
    """Read BioVision Hierarchical (.bvh) file."""

    def __init__(self, filename):

        self.filename = filename
        # A list of unprocessed tokens (strings)
        self._token_list = []
        # The current line number
        self._line_num = 0

        # Root node
        self.root = None
        self._node_stack = []

        # Total number of channels, will be used for on_motion()
        self.num_channels = 0

        self.frames = 0
        self.frame_time = .033333333
        self.frame_values = []

        self._file_handle = None

    def read(self):
        """Read the entire file and return Skeleton."""
        with open(self.filename, 'r') as self._file_handle:
            self.read_hierarchy()
            self.read_motion()
        rig = Skeleton(self.root, keyframes=np.array(self.frame_values), frames=self.frames, frame_time=self.frame_time)
        return rig

    def read_hierarchy(self):
        """Read the skeleton hierarchy."""
        tok = self.token()
        if tok != "HIERARCHY":
            raise SyntaxError("Syntax error in line %d: 'HIERARCHY' expected, "
                              "got '%s' instead" % (self._line_num, tok))
        tok = self.token()
        if tok != "ROOT":
            raise SyntaxError("Syntax error in line %d: 'ROOT' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        self.root = Node(root=True)
        self._node_stack.append(self.root)
        self.read_node()

    def read_node(self):
        """Read the data for a node."""

        # Read the node name (or the word 'Site' if it was a 'End Site' node)
        name = self.token()
        self._node_stack[-1].name = name

        tok = self.token()
        if tok != "{":
            raise SyntaxError("Syntax error in line %d: '{' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        while 1:
            tok = self.token()
            if tok == "OFFSET":
                x = self.float_token()
                y = self.float_token()
                z = self.float_token()
                self._node_stack[-1].offset = (x, y, z)
                self._node_stack[-1].transmat[0, 3] = x
                self._node_stack[-1].transmat[1, 3] = y
                self._node_stack[-1].transmat[2, 3] = z
            elif tok == "CHANNELS":
                n = self.int_token()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                   "Xrotation", "Yrotation", "Zrotation"]:
                        raise SyntaxError("Syntax error in line %d: Invalid "
                                          "channel name: '%s'"
                                          % (self._line_num, tok))
                    channels.append(tok)
                self.num_channels += len(channels)
                self._node_stack[-1].channels = channels
            elif tok == "JOINT":
                node = Node()
                node.parent = self._node_stack[-1]
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "End":
                node = Node()
                node.parent = self._node_stack[-1]
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "}":
                if self._node_stack[-1].is_end_site:
                    self._node_stack[-1].name = self._node_stack[-1].parent.name + "End"
                self._node_stack.pop()
                break
            else:
                raise SyntaxError("Syntax error in line %d: Unknown "
                                  "keyword '%s'" % (self._line_num, tok))

    def read_motion(self):
        """Read the motion samples."""
        # No more tokens (i.e. end of file)? Then just return
        try:
            tok = self.token()
        except StopIteration:
            return

        if tok != "MOTION":
            raise SyntaxError("Syntax error in line %d: 'MOTION' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        # Read the number of frames
        tok = self.token()
        if tok != "Frames:":
            raise SyntaxError("Syntax error in line %d: 'Frames:' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        self.frames = self.int_token()

        # Read the frame time
        tok = self.token()
        if tok != "Frame":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' "
                              "expected, got '%s' instead"
                              % (self._line_num, tok))
        tok = self.token()
        if tok != "Time:":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' "
                              "expected, got 'Frame %s' instead"
                              % (self._line_num, tok))

        self.frame_time = self.float_token()

        # Read the channel values
        for i in range(self.frames):
            s = self.read_line()
            a = s.split()
            if len(a) != self.num_channels:
                raise SyntaxError("Syntax error in line %d: %d float values "
                                  "expected, got %d instead"
                                  % (self._line_num, self.num_channels,
                                     len(a)))
            # In Python 3 map returns map-object, not a list. Can't slice.
            self.frame_values.append(list(map(lambda x: float(x), a)))

            # Implementation of tokenizer

    def int_token(self):
        """Return the next token which must be an int. """
        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Integer expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def float_token(self):
        """Return the next token which must be a float."""
        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Float expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def token(self):
        """Return the next token."""

        # Are there still some tokens left? then just return the next one
        if self._token_list:
            tok = self._token_list[0]
            self._token_list = self._token_list[1:]
            return tok

        # Read a new line
        s = self.read_line()
        self.create_tokens(s)
        return self.token()

    def read_line(self):
        """Return the next line.
        Empty lines are skipped. If the end of the file has been
        reached, a StopIteration exception is thrown.  The return
        value is the next line containing data (this will never be an
        empty string).
        """
        # Discard any remaining tokens
        self._token_list = []
        # Read the next line
        while 1:
            s = self._file_handle.readline()
            self._line_num += 1
            if s == "":
                raise StopIteration
            return s

    def create_tokens(self, s):
        """Populate the token list from the content of s."""
        s = s.strip()
        a = s.split()
        self._token_list = a


#######################################
# NON-CLASS FUNCTIONS START HERE
#######################################


def process_bvhkeyframe(worldpos, quaternion, keyframe, joint, frame):
    """Calculate the global 3D coordinate for the joint"""

    # We have to build up drotmat one rotation value at a time so that
    # we get the matrix multiplication order correct.
    rotmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                       [0., 0., 1., 0.], [0., 0., 0., 1.]])

    counter = 0

    # Suck in as many values off the front of "keyframe" as we need
    # to populate this joint's channels.  The meanings of the keyvals
    # aren't given in the keyframe itself; their meaning is specified
    # by the channel names.
    has_xrot = False
    has_yrot = False
    has_zrot = False
    for channel in joint.channels:
        keyval = keyframe[counter]
        if channel == "Xposition":
            xpos = keyval
        elif channel == "Yposition":
            ypos = keyval
        elif channel == "Zposition":
            zpos = keyval
        elif channel == "Xrotation":
            has_xrot = True
            xrot = keyval
            theta = radians(xrot)
            mycos = cos(theta)
            mysin = sin(theta)
            rotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]])
            rotmat2[1, 1] = mycos
            rotmat2[1, 2] = -mysin
            rotmat2[2, 1] = mysin
            rotmat2[2, 2] = mycos
            rotmat = np.matmul(rotmat, rotmat2)

        elif channel == "Yrotation":
            has_yrot = True
            yrot = keyval
            theta = radians(yrot)
            mycos = cos(theta)
            mysin = sin(theta)
            rotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]])
            rotmat2[0, 0] = mycos
            rotmat2[0, 2] = mysin
            rotmat2[2, 0] = -mysin
            rotmat2[2, 2] = mycos
            rotmat = np.matmul(rotmat, rotmat2)

        elif channel == "Zrotation":
            has_zrot = True
            zrot = keyval
            theta = radians(zrot)
            mycos = cos(theta)
            mysin = sin(theta)
            rotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]])
            rotmat2[0, 0] = mycos
            rotmat2[0, 1] = -mysin
            rotmat2[1, 0] = mysin
            rotmat2[1, 1] = mycos
            rotmat = np.matmul(rotmat, rotmat2)
        else:
            raise SyntaxError("Syntax error in process_bvhkeyframe: Invalid channel"
                              " name ", channel)
        counter += 1

    # End "for channel..."
    if has_xrot and has_yrot and has_zrot:  # End sites don't have rotations.
        joint.rot[frame] = (xrot, yrot, zrot)
        quaternion[frame][joint.name] = q.euler_to_quaternion(np.array(joint.rot[frame]), 'zyx')

    if joint.is_root:  # If the joint is the root.
        # Build a translation matrix for this keyframe.
        joint.transmat[0, 3] = xpos
        joint.transmat[1, 3] = ypos
        joint.transmat[2, 3] = zpos

    # At this point we should have computed:
    #  transmat  (computed previously in process_bvhnode subroutine and root in previous ste[])
    #  rotmat
    # We now have enough to compute joint.trtr and also to convert
    # the position of this joint (vertex) to worldspace.
    #
    # For the non-hips case, we assume that our parent joint has already
    # had its trtr matrix appended to the end of self.trtr[]
    # and that the appropriate matrix from the parent is the LAST item
    # in the parent's trtr[] matrix list.
    #
    # Worldpos of the current joint is localtoworld = TRTR...T*[0,0,0,1]
    #   which equals parent_trtr * T*[0,0,0,1]
    # In other words, the rotation value of a joint has no impact on
    # that joint's position in space, so drotmat doesn't get used to
    # compute worldpos in this routine.
    #
    # However we don't pass localtoworld down to our child -- what
    # our child needs is trtr = TRTRTR...TR
    #
    # The code below attempts to optimize the computations so that we
    # compute localtoworld first, then trtr.

    if not joint.is_root:  # Not hips
        # parent_trtr = joint.parent.trtr[-1]  # Last entry from parent
        parent_global_transform = joint.parent.global_transform  # Dictionary-based rewrite
        localtoworld = np.matmul(parent_global_transform, joint.transmat)
    else:  # Hips
        localtoworld = joint.transmat

    joint.global_transform = np.matmul(localtoworld, rotmat)

    # worldpos[joint.name][time] = [localtoworld[0, 3], localtoworld[1, 3], localtoworld[2, 3]]

    worldpos[frame][joint.name] = [localtoworld[0, 3], localtoworld[1, 3], localtoworld[2, 3]]

    newkeyframe = keyframe[counter:]  # Slices from counter+1 to end
    for child in joint.children:
        # Here's the recursion call.  Each time we call process_bvhkeyframe,
        # the returned value "newkeyframe" should shrink due to the slicing
        # process
        newkeyframe = process_bvhkeyframe(worldpos, quaternion, newkeyframe, child, frame)
        # if newkeyframe == 0:  # If retval = 0
        #     print("Passing up fatal error in process_bvhkeyframe")
        #     return 0
    return newkeyframe


def remove_joints(rig, joints_list):

    # remove the node in the list except the root
    def remove_node(node, remove_list):
        new_remove_list = remove_list[:]
        for remove_joint in remove_list:
            children_name = []
            for child in node.children:
                children_name.append(child.name)
            if remove_joint in children_name:
                temp_node = node.children.pop(children_name.index(remove_joint))
                children_name.remove(remove_joint)
                if not temp_node.is_end_site:
                    node.children = node.children + temp_node.children
                    for child in node.children:
                        child.parent = node
                if node.is_end_site:
                    new_remove_list.append(node.name)
                    node.name = node.name + 'End'

        for child in node.children:
            new_remove_list = remove_node(child, new_remove_list)

        return new_remove_list

    new_joints_list = remove_node(rig.root, joints_list)

    for frame in rig.quaternion:
        for joint in new_joints_list:
            if joint in rig.quaternion[frame].keys():
                del rig.quaternion[frame][joint]

    for frame in rig.worldpos:
        for joint in joints_list:
            if joint in rig.worldpos[frame].keys():
                del rig.worldpos[frame][joint]


def data_store(rigs, filename='cmu_test.npz'):
    locomotion_pos = []
    locomotion_rot = []
    locomotion_styles = []
    locomotion_actions = []
    locomotion_skeletons = []
    counters = {}
    for rig in rigs:
        pos = None
        rot = None
        for frame in rig.worldpos:
            if pos is not None:
                pos = np.dstack([pos, np.array(list(rig.worldpos[frame].values()))])
            else:
                pos = np.array(list(rig.worldpos[frame].values()))
        for frame in rig.quaternion:
            if rot is not None:
                rot = np.dstack([rot, np.array(list(rig.quaternion[frame].values()))])
            else:
                rot = np.array(list(rig.quaternion[frame].values()))
        pos = np.moveaxis(pos, 2, 0)
        rot = np.moveaxis(rot, 2, 0)
        locomotion_pos.append(pos)
        locomotion_rot.append(q.qfix(rot))
        locomotion_styles.append('unknown')
        locomotion_actions.append('walk')
        locomotion_skeletons.append(rig.root)
    output_file_path = filename
    np.savez_compressed(output_file_path,
                        worldpos=locomotion_pos,
                        rotations=locomotion_rot,
                        styles=locomotion_styles,
                        actions=locomotion_actions,
                        skeletons=locomotion_skeletons)
    return
