import torch
import torch.nn as nn
from config.base_config import Config
from modules.CAttention import CAM
from modules.loss import KL

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config

        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError
        embed_dim = self.config.embed_dim
        self.cf_c_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.cf_f_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.cam_sf = CAM(embed_dim=embed_dim, dropout=0.3)
        self.cam_sc = CAM(embed_dim=embed_dim, dropout=0.3)

        self.loss_kl = KL()

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        v_feat = self.clip.get_image_features(video_data)
        f_feat = v_feat.reshape(batch_size, self.config.num_frames, -1)
        s_feat = self.clip.get_text_features(**data['cap'])

        c_feat_list = []
        for i in range(self.config.num_captions):
            c_feat = self.clip.get_text_features(**data['seq_cap'][i])
            c_feat_list.append(c_feat)
        c_feat = torch.stack(c_feat_list, dim=1)

        if is_train:
            # init params:
            a, d = s_feat.size(0), s_feat.size(1)
            b, c, d = c_feat.size(0), c_feat.size(1), c_feat.size(2)
            b, f, d = f_feat.size(0), f_feat.size(1), f_feat.size(2),

            ########## Step-I: Sort ##########
            # sims_sf = torch.einsum("ad,bfd->abf", [self.norm(s_feat), self.norm(f_feat)])
            # sims_sf = sims_sf.diagonal(dim1=0, dim2=1).transpose(0, 1)
            # _, f_new_idx = torch.topk(sims_sf, k=f, dim=-1, largest=True)
            # f_feat = f_feat[torch.arange(b)[:, None], f_new_idx, :]
            # sims_sc = torch.einsum("ad,bcd->abc", [self.norm(s_feat), self.norm(c_feat)])
            # sims_sc = sims_sc.diagonal(dim1=0, dim2=1).transpose(0, 1)
            # _, c_new_idx = torch.topk(sims_sc, k=c, dim=-1, largest=True)
            # c_feat = c_feat[torch.arange(b)[:, None], c_new_idx, :]

            ########## Step-II: Interaction ##########
            # <c_feat, f_feat>
            c_w = torch.softmax(self.cf_c_feat_w(c_feat).squeeze(-1), dim=-1)
            f_w = torch.softmax(self.cf_f_feat_w(f_feat).squeeze(-1), dim=-1)
            sims_cf = torch.einsum("acd,bfd->abcf", [self.norm(c_feat), self.norm(f_feat)])
            sims_c2f, _ = sims_cf.max(dim=-1)
            sims_c2f = torch.einsum('abc,ac->ab', [sims_c2f, c_w])
            sims_f2c, _ = sims_cf.max(dim=-2)
            sims_f2c = torch.einsum('abf,bf->ab', [sims_f2c, f_w])
            sims_cf = (sims_c2f + sims_f2c) / 2.0

            # <s_feat, f_feat>
            f_feat_agg = self.cam_sf(s_feat, f_feat)
            sims_sf = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(f_feat_agg)])
            # <s_feat, c_feat>
            c_feat_agg = self.cam_sc(s_feat, c_feat)
            sims_sc = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(c_feat_agg)])

            ########## Step-III: KL Loss ##########
            loss_kl = (self.loss_kl(sims_sf, sims_cf) + self.loss_kl(sims_sf, sims_cf.T) +
                       self.loss_kl(sims_sc, sims_cf) + self.loss_kl(sims_sc, sims_cf.T)) / 4.0

            return sims_cf, sims_sf, sims_sc, loss_kl
        else:
            return s_feat, c_feat, f_feat

    def norm(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def get_similarity_logits(self, s_feat, c_feat, f_feat):
        f_feat_agg = self.cam_sf(s_feat, f_feat)
        sims_sf = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(f_feat_agg)])
        # <s_feat, c_feat>
        c_feat_agg = self.cam_sc(s_feat, c_feat)
        sims_sc = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(c_feat_agg)])

        sims = sims_sf + sims_sc

        return sims