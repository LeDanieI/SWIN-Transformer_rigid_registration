import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

from model.SwinTransformer import SwinTransformer

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

class SpatialTransformerLayer(nn.Module):
    """
    N-D Spatial Transformer Layer
    !! Nnote this is not a Vision transformer but an image interpolator
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, T):
        # rigid grid
        grid = nnf.affine_grid(T, src.size(), align_corners=True)
        # shape = src.shape[2:]
        # for i in range(len(shape)):
        #     grid[..., i] = ((grid[..., i] + 1) / 2)  * (shape[i] - 1)
        #     grid[..., i] = 2 * (grid[..., i] / (shape[i] - 1) - 0.5)
        return nnf.grid_sample(src, grid, align_corners=True, mode=self.mode)


class ScaledTanH(nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def forward(self, input):
        return torch.tanh(input) * self.scaling

    def __repr__(self):
        return self.__class__.__name__ + "(" + "scaling = " + str(self.scaling) + ")"


class RigidRegistrationHead(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.max_rotation = 0.5 * np.pi
        self.translation = nn.Linear(in_channels, 3)
        self.euler_angles = nn.Sequential(
            nn.Linear(in_channels, 3),
            ScaledTanH(self.max_rotation),
        )
        # init rigid layer (identity transform)
        self.translation.weight.data.zero_()
        self.translation.bias.data.zero_()
        self.euler_angles[0].weight.data.zero_()
        self.euler_angles[0].bias.data.zero_()

class RegTransformer(nn.Module):
    def __init__(self, config):
        super(RegTransformer, self).__init__()
        # initialize
        self.config = config
        embed_dim = config.embed_dim
        patch_size = config.patch_size
        self.swin_transformer = SwinTransformer(patch_size=config.patch_size,
                                                in_chans=config.in_chans,
                                                embed_dim=config.embed_dim,
                                                depths=config.depths,
                                                num_heads=config.num_heads,
                                                window_size=config.window_size,
                                                mlp_ratio=config.mlp_ratio,
                                                qkv_bias=config.qkv_bias,
                                                drop_rate=config.drop_rate,
                                                drop_path_rate=config.drop_path_rate,
                                                ape=config.ape,
                                                spe=config.spe,
                                                rpe=config.rpe,
                                                patch_norm=config.patch_norm,
                                                use_checkpoint=config.use_checkpoint,
                                                out_indices=config.out_indices,
                                                pat_merg_rf=config.pat_merg_rf,
                                                )

        channel = embed_dim*config.img_size[0]*config.img_size[1]*config.img_size[2]//(4**3)
        self.reg_head = RigidRegistrationHead(in_channels=channel)
        self.stl = SpatialTransformerLayer(config.img_size)
        self.stl_binary = SpatialTransformerLayer(config.img_size, mode='nearest')

    def forward(self, moving, fixed):
        # concatenate the moving and fixed images and forward pass through SwinTransformer (encoder)
        x = torch.cat((moving, fixed), dim=1)
        out_feats = self.swin_transformer(x)
        # print(out_feats)
        # transform features into rigid params to obtain the transformation matrix (T)
        enc_output_flatten = out_feats[0].view(1, -1)
        # print(enc_output_flatten.shape)
        trans = self.reg_head.translation(enc_output_flatten)
        # print(trans)
        angles = self.reg_head.euler_angles(enc_output_flatten)
        rot_mat = euler_angles_to_matrix(euler_angles=angles, convention="XYZ")
        # print(rot_mat)
        T = torch.cat([rot_mat.squeeze(), trans.squeeze().view(3, 1)], axis=1)
        
        T = T.view(-1, 3, 4)
        # print(T)
        moving_warped = self.stl(moving, T)
        return moving_warped, T, angles,trans

