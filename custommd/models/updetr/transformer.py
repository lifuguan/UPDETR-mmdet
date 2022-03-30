'''
Author: your name
Date: 2022-03-28 22:41:19
LastEditTime: 2022-03-28 22:53:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/custommd/models/updetr/transformer.py
'''

import torch
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

@TRANSFORMER.register_module()
class CustomTransformer(BaseModule):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(CustomTransformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, attn_masks=None):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)

        # to align with UP-DETR, we add self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) here.
        # [num_query, dim] -> [num_query, bs, dim]
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  
        
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory