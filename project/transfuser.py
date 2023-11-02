"""
Implements the TransFuser vision backbone.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
import huggingface_hub
import copy


class TransfuserBackbone(nn.Module):
  """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.image_encoder = timm.create_model("regnety_032", pretrained=True, features_only=True).to(self.config.device)
    self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors)).to(self.config.device)

    in_channels = 1
    self.lidar_encoder = timm.create_model("regnety_032",
                                            pretrained=False,
                                            in_chans=in_channels,
                                            features_only=True).to(self.config.device)

    self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1).to(self.config.device)
    self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors)).to(self.config.device)
    lidar_time_frames = [1, 1, 1, 1]

    self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1).to(self.config.device)
    start_index = 0
    # Some networks have a stem layer
    self.transformers = nn.ModuleList([
        GPT(n_embd=self.image_encoder.feature_info.info[0]['num_chs'],
            config=self.config,
            lidar_time_frames=lidar_time_frames[i]) for i in range(4)
    ])

    self.lidar_channel_to_img = nn.ModuleList([
        nn.Conv2d(self.lidar_encoder.feature_info.info[0]['num_chs'],
                self.image_encoder.feature_info.info[0]['num_chs'],
                kernel_size=1) for i in range(4)
    ])
    self.img_channel_to_lidar = nn.ModuleList([
        nn.Conv2d(self.image_encoder.feature_info.info[0]['num_chs'],
                self.lidar_encoder.feature_info.info[0]['num_chs'],
                kernel_size=1) for i in range(4)
    ])

    self.num_image_features = self.image_encoder.feature_info.info[3]['num_chs']
    # Typical encoders down-sample by a factor of 32
    self.perspective_upsample_factor = self.image_encoder.feature_info.info[
        start_index + 3]['reduction'] // self.config.perspective_downsample_factor

    # Number of features the encoder produces.
    self.num_features = self.image_encoder.feature_info.info[3]['num_chs'] + \
                        self.lidar_encoder.feature_info.info[3]['num_chs']

  def forward(self, image, lidar):
    '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        '''
    image_features = normalize_imagenet(image)
    lidar_features = lidar

    # Generate an iterator for all the layers in the network that one can loop through.
    image_layers = iter(self.image_encoder.items())
    lidar_layers = iter(self.lidar_encoder.items())

    # Loop through the 4 blocks of the network.
    for i in range(4):
        image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)


    image_feature_grid = None

    image_features = self.global_pool_img(image_features)
    image_features = torch.flatten(image_features, 1)
    lidar_features = self.global_pool_lidar(lidar_features)
    lidar_features = torch.flatten(lidar_features, 1)

    fused_features = torch.cat((image_features, lidar_features), dim=1)


    return fused_features, image_feature_grid

  def forward_layer_block(self, layers, return_layers, features):
    """
    Run one forward pass to a block of layers from a TIMM neural network and returns the result.
    Advances the whole network by just one block
    :param layers: Iterator starting at the current layer block
    :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
    :param features: Input features
    :return: Processed features
    """
    for name, module in layers:
      features = module(features)
      if name in return_layers:
        break
    return features

  def fuse_features(self, image_features, lidar_features, layer_idx):
    """
    Perform a TransFuser feature fusion block using a Transformer module.
    :param image_features: Features from the image branch
    :param lidar_features: Features from the LiDAR branch
    :param layer_idx: Transformer layer index.
    :return: image_features and lidar_features with added features from the other branch.
    """
    image_embd_layer = self.avgpool_img(image_features)
    lidar_embd_layer = self.avgpool_lidar(lidar_features)

    lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

    image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)

    lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

    image_features_layer = F.interpolate(image_features_layer,
                                         size=(image_features.shape[2], image_features.shape[3]),
                                         mode='bilinear',
                                         align_corners=False)

    lidar_features_layer = F.interpolate(lidar_features_layer,
                                        size=(lidar_features.shape[2], lidar_features.shape[3]),
                                        mode='bilinear',
                                        align_corners=False)

    image_features = image_features + image_features_layer
    lidar_features = lidar_features + lidar_features_layer

    return image_features, lidar_features


class GPT(nn.Module):
  """  the full GPT language backbone, with a context size of block_size """

  def __init__(self, n_embd, config, lidar_time_frames):
    super().__init__()
    self.config = config
    self.n_embd = n_embd
    # We currently only support seq len 1
    self.seq_len = 1
    self.lidar_seq_len = self.config.lidar_seq_len
    
    self.lidar_time_frames = lidar_time_frames

    # positional embedding parameter (learnable), image + lidar
    self.pos_emb = nn.Parameter(
        torch.zeros(
            1, self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors +
            lidar_time_frames * self.config.lidar_vert_anchors * self.config.lidar_horz_anchors, self.n_embd)).to(self.config.device)

    # transformer
    self.blocks = nn.Sequential(*[
        Block(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop).to(self.config.device)
        for layer in range(config.n_layer)
    ])

    # decoder head
    self.ln_f = nn.LayerNorm(n_embd)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

  def forward(self, image_tensor, lidar_tensor):
    """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
        """

    bz = lidar_tensor.shape[0]

    lidar_h, lidar_w = lidar_tensor.shape[2:4]
    img_h, img_w = image_tensor.shape[2:4]

    assert self.seq_len == 1
    image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
    lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

    token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

    x = self.blocks(x)  # (B, an * T, C)
    x = self.ln_f(x)  # (B, an * T, C)

    image_tensor_out = x[:, :self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :].view(
        bz * self.seq_len, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()

    lidar_tensor_out = x[:, self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors:, :].view(
        bz, lidar_h, lidar_w, -1).permute(0, 3, 1, 2).contiguous()

    return image_tensor_out, lidar_tensor_out


class SelfAttention(nn.Module):
  """
    A vanilla multi-head masked self-attention layer with a projection at the
    end.
    """

  def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
    super().__init__()
    assert n_embd % n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(n_embd, n_embd)
    self.query = nn.Linear(n_embd, n_embd)
    self.value = nn.Linear(n_embd, n_embd)
    # regularization
    self.attn_drop = nn.Dropout(attn_pdrop)
    self.resid_drop = nn.Dropout(resid_pdrop)
    # output projection
    self.proj = nn.Linear(n_embd, n_embd)
    self.n_head = n_head

  def forward(self, x):
    b, t, c = x.size()

    # calculate query, key, values for all heads in batch and move head
    # forward to be the batch dim
    k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
    q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
    v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

    # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    att = self.attn_drop(att)
    y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
    y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

    # output projection
    y = self.resid_drop(self.proj(y))
    return y


class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
    super().__init__()
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, block_exp * n_embd),
        nn.ReLU(True),  # changed from GELU
        nn.Linear(block_exp * n_embd, n_embd),
        nn.Dropout(resid_pdrop),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))

    return x



def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
        Args:
            x (tensor): input images
        """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x
