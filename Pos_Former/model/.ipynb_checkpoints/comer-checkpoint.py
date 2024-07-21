from typing import List ,Tuple
import cv2, random, torchvision, time
import numpy as np
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
import os
from comer_pos_version1.utils.utils import Hypothesis

from .decoder import Decoder , PosDecoder
from .encoder import Encoder
from comer_pos_version1.datamodule import vocab , label_make_muti

class CoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        # self.trainer = pl.Trainer()
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.posdecoder = PosDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage
        
        )
        # self.step = 0
        self.save_path = 'attn_PosFormer'

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor,FloatTensor,FloatTensor]:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)
        
        tgt_list=tgt.cpu().numpy().tolist()
        muti_labels=label_make_muti.tgt2muti_label(tgt_list)
        muti_labels_tensor=torch.FloatTensor(muti_labels)   #[2b,l,5]
        muti_labels_tensor=muti_labels_tensor.cuda()
        
        out, cls_attn = self.decoder(feature, mask, tgt)
        out_layernum , out_pos, pos_attn=self.posdecoder(feature, mask,tgt,muti_labels_tensor)
        #draw attn
        nc, nt, nl = cls_attn.shape
        # nt_ = pos_attn.shape[1]
        word=tgt[0].tolist()
        sp=os.path.join(self.save_path, vocab.indices2label(word[1:]))
        if not os.path.exists(sp):
            os.mkdir(sp)
        # self.save_counter+=1
        try:
            single_nt = word.index(0) - 1
        except:
            single_nt=nt
        image0 = img[0]
        # overlaps = []
        ABI_scores = cls_attn.reshape(-1, 8, nt, feature.shape[1], feature.shape[2]).mean(1)
        attn_scores = ABI_scores[0].detach().cpu().numpy()
        image_numpy = image0.detach().cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        for t in range(single_nt): #nt
            att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
            att_map = cv2.resize(att_map, (image0.shape[2], image0.shape[1]))  # [H, W]
            att_map = (att_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]

            overlap = cv2.addWeighted( x, 0.6,heatmap, 0.4, 0, dtype=cv2.CV_32F)
            try:
                word_to_be_pred=vocab.indices2label([word[t+1]])
            except:
                word_to_be_pred='eos'
            cv2.imwrite(os.path.join(sp, f"attn_{t}_{word_to_be_pred}.png"), overlap)
        # for i in range(len(tgt.tolist())):
        #     try:
        #         word=tgt[i].tolist()
        #         word=word[:15]
        #     except:
        #         continue
        #     # if word == [1, 53, 110, 13, 82, 110, 16, 112, 112, 110, 14, 6, 12, 112]:
        #     if word == [1, 62, 63, 83, 110, 107, 71, 53, 110, 12, 112, 110, 15, 112, 112]:
        #         self.save_counter+=1
        #         try:
        #             single_nt = tgt[i].tolist().index(0) - 1
        #         except:
        #             single_nt=nt
        #         image0 = img[i]
        #         overlaps = []
        #         ABI_scores = cls_attn.reshape(-1, 8, nt, feature.shape[1], feature.shape[2]).mean(1)
        #         attn_scores = ABI_scores[i].detach().cpu().numpy()
        #         image_numpy = image0.detach().cpu().float().numpy()
        #         if image_numpy.shape[0] == 1:
        #             image_numpy = np.tile(image_numpy, (3, 1, 1))
        #         x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        #         for t in range(single_nt): #nt
        #             att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
        #             att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
        #             att_map = cv2.resize(att_map, (image0.shape[2], image0.shape[1]))  # [H, W]
        #             att_map = (att_map * 255).astype(np.uint8)
        #             heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]

        #             overlap = cv2.addWeighted( x, 0.6,heatmap, 0.4, 0, dtype=cv2.CV_32F)
        #             cv2.imwrite(os.path.join(self.save_path, f"attention_map_{self.save_counter}_{t}.png"), overlap)
        # self.step += 1
        return out, out_layernum, out_pos   # [2b,l,vocab_size], [2b,l,5] and[2b,l,6]

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        # beam_size = 1
        # print('???????????????????????????????')
        # print(beam_size)
        seq_out= self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )

        return seq_out