import torch

class JEPAVoxelAdapter:
    """
    Converts BEVCar voxel outputs into JEPA-compatible tokens.
    Keeps BEVCar untouched.
    """

    def __init__(self):
        pass

    def to_tokens(self, vf, vc, nv):
        """
        vf: [B, N, P, F] or [B, N, F]
        vc: [B, N, 3]
        nv: [B]
        """

        # Case 1: already clean
        if vf.dim() == 3:
            return vf, vc, nv

        # Case 2: BEVCar voxel structure [B, N, P, F]
        B, N, P, F = vf.shape

        # simple aggregation (mean over points)
        vf_tokens = vf.mean(dim=2)  # -> [B, N, F]

        return vf_tokens, vc, nv