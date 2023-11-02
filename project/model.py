"""
The main model structure
"""
import numpy as np
from pathlib import Path
from transfuser import TransfuserBackbone
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy
import math
import os


class LidarCenterNet(nn.Module):
  """
  The main model class. It can run all model configurations.
  """

  def __init__(self):
    super().__init__()

    self.config = ModelConfig()
    self.speed_histogram = []
    self.make_histogram = int(os.environ.get('HISTOGRAM', 0))

    self.backbone = TransfuserBackbone(self.config)
    
    self.wp_query = nn.Parameter(torch.zeros(1, self.config.pred_len, self.config.gru_input_size))

  def forward(self, rgb, lidar_bev, target_point):
    bs = rgb.shape[0]

    bev_feature_grid, fused_features, image_feature_grid = self.backbone(rgb, lidar_bev)

    joined_wp_features = self.join(self.wp_query.repeat(bs, 1, 1), fused_features)
    pred_wp = self.wp_decoder(joined_wp_features, target_point)

    return pred_wp

  def compute_loss(self, pred_wp, waypoint_label):
    loss_wp = torch.mean(torch.abs(pred_wp - waypoint_label))
    return loss_wp

class ModelConfig():
    def __init__(self):

        self.device = "cuda"
        
        self.gru_input_size = 256
        self.pred_len = 8
        self.img_vert_anchors = 24
        self.img_horz_anchors = 24
        self.lidar_vert_anchors = 24
        self.lidar_horz_anchors = 24
        self.perspective_downsample_factor = 1
        self.lidar_seq_len = 1

        #transformer shit
        self.n_head = 4
        self.block_exp = 4
        self.attn_pdrop = 0
        self.resid_pdrop = 0
        self.n_layer = 4
        self.gpt_linear_layer_init_mean = 0
        self.gpt_linear_layer_init_std = 1
        self.gpt_layer_norm_init_weight = 1


class GRUWaypointsPredictorTransFuser(nn.Module):
  """
  The waypoint GRU used in TransFuser.
  It enters the target point as input.
  The hidden state is initialized with the scene features.
  The input is autoregressive and starts either at 0 or learned.
  """

  def __init__(self, pred_len, hidden_size, target_point_size):
    super().__init__()
    self.wp_decoder = nn.GRUCell(input_size=2 + target_point_size, hidden_size=hidden_size)
    self.output = nn.Linear(hidden_size, 2)
    self.prediction_len = pred_len

  def forward(self, z, target_point):
    output_wp = []

    x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

    target_point = target_point.clone()

    # autoregressive generation of output waypoints
    for _ in range(self.prediction_len):
        x_in = torch.cat([x, target_point], dim=1)
        z = self.wp_decoder(x_in, z)
        dx = self.output(z)
        x = dx + x

        output_wp.append(x)

    pred_wp = torch.stack(output_wp, dim=1)

    return pred_wp


class PositionEmbeddingSine(nn.Module):
  """
  Taken from InterFuser
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature

    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, tensor):
    x = tensor
    bs, _, h, w = x.shape
    not_mask = torch.ones((bs, h, w), device=x.device)
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
