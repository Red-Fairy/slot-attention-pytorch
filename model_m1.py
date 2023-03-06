from turtle import position
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderPosEmbedding(nn.Module):
    def __init__(self, dim, resolution, hidden_dim=128, scale_factor=5):
        super().__init__()
        self.grid = build_grid(resolution, single=True) # (1, h, w, 2)
        self.grid_embed = nn.Linear(2, dim, bias=True)
        self.input_to_k = nn.Linear(dim, dim, bias=False)
        self.input_to_v = nn.Linear(dim, dim, bias=False)

        self.MLP = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.scale_factor = scale_factor

    def apply_rel_position_scale(self, grid, position, scale):
        """
        grid: (1, h, w, 2)
        position (batch, number_slots, 2)
        scale (batch, number_slots, 2)
        """
        b, n, _ = position.shape
        h, w = grid.shape[1:3]
        grid = grid.view(1, 1, h, w, 2)
        grid = grid.repeat(b, n, 1, 1, 1)
        position = position.view(b, n, 1, 1, 2)
        # position = position.repeat(1, 1, h, w, 1)
        scale = scale.view(b, n, 1, 1, 2)
        return ((grid - position) / (scale * self.scale_factor + 1e-8)) # (b, n, h, w, 2)

    def forward(self, x, position_latent=None, scale_latent=None):

        k, v = self.input_to_k(x), self.input_to_v(x) # (b, h*w, d)
        k, v = k.unsqueeze(1), v.unsqueeze(1) # (b, 1, h*w, d)

        if position_latent is not None or scale_latent is not None:
            rel_grid = self.apply_rel_position_scale(self.grid, position_latent, scale_latent)
        else:
            rel_grid = self.grid
        rel_grid = rel_grid.flatten(-3, -2)

        # rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, n, h*w, 4)

        grid_embed = self.grid_embed(rel_grid) # (b, n, h*w, d)

        k, v = k + grid_embed, v + grid_embed
        k, v = self.MLP(k), self.MLP(v)

        return k, v # (b, n, h*w, d)

class DecoderPosEmbedding(nn.Module):
    def __init__(self, resolution=(8, 8), hidden_dim=128, scale_factor=5):
        super().__init__()
        self.grid_embed = nn.Linear(2, hidden_dim, bias=True)
        self.grid = build_grid(resolution, single=True) # (1, h, w, 2)
        self.scale_factor = scale_factor

    def apply_rel_position_scale(self, grid, position, scale):
        """
        grid: (1, h, w, 2)
        position (batch*number_slots, 2)
        scale (batch*number_slots, 2)
        """
        h, w = grid.shape[1:3]
        bns = position.shape[0]
        grid = grid.expand(bns, h, w, 2)
        position = position.unsqueeze(1).unsqueeze(1).expand(bns, h, w, 2) # bns, h, w, 2
        scale = scale.unsqueeze(1).unsqueeze(1).expand(bns, h, w, 2)
        return ((grid - position) / (scale * self.scale_factor + 1e-8))

    def forward(self, x, position_latent, scale_latent):
        '''
        x: (b*n_s, h, w, d)
        position_latent: (b, n_s, 2)
        '''
        rel_grid = self.apply_rel_position_scale(self.grid, position_latent, scale_latent) # (bns, h, w, 2)
        # rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1) # (bns, h, w, 4)
        grid_embed = self.grid_embed(rel_grid) # (bns, h, w, d)
        # print(x.shape, grid_embed.shape)
        
        return x + grid_embed


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, resolution=(128,128)):
        '''
        num_slots: number of slots
        dim: dimension of slots (slots: K x dim, omit batch size)
        '''
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # intialize slots
        # positional latent in U[-1, 1]
        self.position_latent = nn.Parameter(torch.rand(1, num_slots, 2) * 2 - 1)
        # scaling latent in N(0.1,0.01)
        self.scale_latent = nn.Parameter(torch.randn(1, num_slots, 2) * 0.01 + 0.1)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        # self.to_k = nn.Linear(dim, dim)
        # self.to_v = nn.Linear(dim, dim)
        
        self.to_kv = EncoderPosEmbedding(dim, resolution, hidden_dim, scale_factor=5)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.grid = build_grid(resolution, single=True).flatten(-3, -2) # (1, h*w, 2)

    def forward(self, inputs, num_slots = None):
        b, hw, d = inputs.shape # b, hw, d
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1) # (b, n_s, dim)
        sigma = self.slots_sigma.expand(b, n_s, -1) # (b, n_s, dim)
        slots = torch.normal(mu, sigma)

        position_latent = self.position_latent.repeat(b, 1, 1) # (b, n_s, 2)
        scale_latent = self.scale_latent.repeat(b, 1, 1) # (b, n_s, 2)

        inputs = self.norm_input(inputs)        

        for it in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # compute k and v in every iteration, because relative position/scale is updated
            # (b, n_s, h*w, d), we have n_s (num_slots) different keys and values
            k, v = self.to_kv(inputs, position_latent, scale_latent)

            dots = torch.einsum('bid,bjnd->bijn', q, k) * self.scale # q: (b, n_s, d), k: (b, n_s, hw, d), output: (b, n_s, n_s, hw)
            # softmax on the slot axis
            attn = dots.softmax(dim=1) + self.eps
            # generate updates, attn: (b, n_s, n_s, hw), v: (b, n_s, hw, d), output: (b, n_s, d)
            updates = torch.einsum('bijn,bjnd->bid', attn, v) # attn: (b, n_s, n_s, hw), v: (b, n_s, hw, d), output: (b, n, d)
            # print(updates.shape)
            # print(slots.shape)

            if it != self.iters - 1:
                slots = self.gru(updates.reshape(b*n_s, d), slots_prev.reshape(b*n_s, d)).reshape(b, n_s, d)
                slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

            # update position and scale
            # first sum over the input dimension, output b, n_s, n_s, h*w
            attn = attn / attn.sum(dim=(-2,-1), keepdim=True) # modified
            grid = self.grid.unsqueeze(1).expand(b, n_s, -1, -1) # (b, n_s, h*w, 2)
            position_latent = torch.einsum('bijk,bjkl->bil', attn, grid) # attn: (b, n, n, h*w), grid: (b, n, h*w, 2), output: (b, n, 2)
            rel_pos = grid - position_latent.unsqueeze(2).expand_as(grid) # grid: (b, n_s, h*w, 2), position_latent: (b, n_s, 1, 2), output: (b, n, h*w, 2)
            scale_latent = torch.sqrt(torch.einsum('bijk,bjkl->bil', attn + self.eps, rel_pos ** 2)) # attn: (b, n, h*w), rel_pos: (b, n, h*w, 2), output: (b, n, 2)

            # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # attn = dots.softmax(dim=1) + self.eps
            # attn = attn / attn.sum(dim=-1, keepdim=True)

            # updates = torch.einsum('bjd,bij->bid', v, attn)

            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )

            # slots = slots.reshape(b, -1, d)
            # slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, position_latent, scale_latent

def build_grid(resolution, single=False):
    ranges = [np.linspace(-1., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    if single:
        return torch.from_numpy(grid).to(device) # (1, h, w, 2)
    else:
        return torch.from_numpy(np.concatenate([grid, -grid], axis=-1)).to(device) # (1, h, w, 4)

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution, flattened=False):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True) 
        self.grid = build_grid(resolution)
        if flattened:
            self.grid = self.grid.view(-1, hidden_size)

    def forward(self, inputs): # input: (b, h, w, c) or (b, h*w, c) for flattened
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        # self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1) # (b, h, w, c)
        # x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2) # (b, h*w, c)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = DecoderPosEmbedding(self.decoder_initial_size, hid_dim)
        self.resolution = resolution

    def forward(self, x, position_latent, scale_latent):
        x = self.decoder_pos(x, position_latent, scale_latent)
        x = x.permute(0,3,1,2) # B, C, H, W
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128,
            resolution = self.resolution)

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots, position_latent, scale_latent = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # position_latent has shape: [batch_size, num_slots, 2]
        # scale_latent has shape: [batch_size, num_slots, 2]

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1)) # (b*n_s, h, w, dim)
        position_latent = position_latent.reshape((-1, position_latent.shape[-1]))
        scale_latent = scale_latent.reshape((-1, scale_latent.shape[-1]))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots, position_latent, scale_latent)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots
