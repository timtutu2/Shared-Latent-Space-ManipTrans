from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from dataset.transform import aa_to_rotmat


class Shadow(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "shadow"
        self.body_names = [
            "palm",
            "ffknuckle",
            "ffproximal",
            "ffmiddle",
            "ffdistal",
            "fftip",
            "lfmetacarpal",
            "lfknuckle",
            "lfproximal",
            "lfmiddle",
            "lfdistal",
            "lftip",
            "mfknuckle",
            "mfproximal",
            "mfmiddle",
            "mfdistal",
            "mftip",
            "rfknuckle",
            "rfproximal",
            "rfmiddle",
            "rfdistal",
            "rftip",
            "thbase",
            "thproximal",
            "thhub",
            "thmiddle",
            "thdistal",
            "thtip",
        ]
        # URDF uses 0-indexed naming (J0 = tip-most, J_max = base-most). Shadow
        # classic convention calls them J1 = tip, J_max+1 = base. We keep the
        # base-to-tip ORDER (same semantics) but rename to URDF convention so
        # the engine's identity-by-name remap can match.
        self.dof_names = [
            "FFJ3", 
            "LFJ4", 
            "MFJ3", 
            "RFJ3", 
            "THJ4", 
            "FFJ2", 
            "LFJ3", 
            "MFJ2", 
            "RFJ2", 
            "THJ3",
            "FFJ1", 
            "LFJ2", 
            "MFJ1", 
            "RFJ1", 
            "THJ2", 
            "FFJ0",         
            "LFJ1",     
            "MFJ0",      
            "RFJ0",      
            "THJ1",  
            "LFJ0",   
            "THJ0"
        ]
        self.hand2dex_mapping = {
            "wrist": ["palm"],
            "thumb_proximal": ["thbase", "thproximal"],  # one-to-many mapping
            "thumb_intermediate": ["thhub", "thmiddle"],
            "thumb_distal": ["thdistal"],
            "thumb_tip": ["thtip"],
            "index_proximal": ["ffknuckle", "ffproximal"],
            "index_intermediate": ["ffmiddle"],
            "index_distal": ["ffdistal"],
            "index_tip": ["fftip"],
            "middle_proximal": ["mfknuckle", "mfproximal"],
            "middle_intermediate": ["mfmiddle"],
            "middle_distal": ["mfdistal"],
            "middle_tip": ["mftip"],
            "ring_proximal": ["rfknuckle", "rfproximal"],
            "ring_intermediate": ["rfmiddle"],
            "ring_distal": ["rfdistal"],
            "ring_tip": ["rftip"],
            "pinky_proximal": ["lfmetacarpal", "lfknuckle", "lfproximal"],
            "pinky_intermediate": ["lfmiddle"],
            "pinky_distal": ["lfdistal"],
            "pinky_tip": ["lftip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "thdistal",
            "ffdistal",
            "mfdistal",
            "rfdistal",
            "lfdistal",
        ]
        self.bone_links = [
            [0, 1],
            [0, 6],
            [0, 12],
            [0, 17],
            [0, 22],
            [2, 3],
            [3, 4],
            [4, 5],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [13, 14],
            [14, 15],
            [15, 16],
            [18, 19],
            [19, 20],
            [20, 21],
            [23, 24],
            [24, 25],
            [25, 26],
            [26, 27],
            # Previously missing: knuckle → proximal connections. For FF/MF/RF
            # these two bodies share a position so the bone is zero-length and
            # invisible, but LF has a real metacarpal → knuckle segment and
            # drawing was disconnected. Adding all of them for completeness.
            [1, 2],    # FF: ffknuckle → ffproximal
            [6, 7],    # LF: lfmetacarpal → lfknuckle
            [12, 13],  # MF: mfknuckle → mfproximal
            [17, 18],  # RF: rfknuckle → rfproximal
            [22, 23],  # TH: thbase → thproximal
        ]
        self.weight_idx = {
            "thumb_tip": [27],
            "index_tip": [5],
            "middle_tip": [16],
            "ring_tip": [21],
            "pinky_tip": [11],
            "level_1_joints": [1, 2, 7, 8, 12, 13, 17, 18, 22, 23],
            "level_2_joints": [3, 4, 6, 9, 10, 14, 15, 19, 20, 24, 25, 26],
        }

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.8
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 80
        self.Ki_pos = 0.005
        self.Kd_pos = 3
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("shadow_rh")
class ShadowRH(Shadow):
    def __init__(self):
        super().__init__()
        self._urdf_path = "data/assets/shadow_hand/shadow_hand_woarm_right/shadow_hand_woarm_right.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        # URDF/USD prefix every link and joint with "r_".
        self.body_names = ["r_" + n for n in self.body_names]
        self.dof_names = ["r_" + n for n in self.dof_names]
        self.hand2dex_mapping = {k: ["r_" + dv for dv in v]
                                 for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["r_" + n for n in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("shadow_lh")
class ShadowLH(Shadow):
    def __init__(self):
        super().__init__()
        self._urdf_path = "data/assets/shadow_hand/shadow_hand_woarm_left/shadow_hand_woarm_left.urdf"
        self.side = "lh"
        self.relative_rotation = aa_to_rotmat(np.array([0, np.pi / 2, 0]))
        # URDF/USD prefix every link and joint with "l_".
        self.body_names = ["l_" + n for n in self.body_names]
        self.dof_names = ["l_" + n for n in self.dof_names]
        self.hand2dex_mapping = {k: ["l_" + dv for dv in v]
                                 for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["l_" + n for n in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_lh"
