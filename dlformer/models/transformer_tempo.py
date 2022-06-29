import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from dlformer.utils.utils import instantiate_from_config
import torch.nn as nn


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class Residual(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        self.conv = nn.Sequential(nn.Conv2d(nf * 3, nf, 3, 1, 1), Normalize(nf), nn.ReLU(),
                                  nn.Conv2d(nf, nf, 3, 1, 1), Normalize(nf), nn.ReLU(),
                                  nn.Conv2d(nf, nf, 3, 1, 1), Normalize(nf), nn.ReLU())

        self.conv1 = nn.Conv2d(nf, nf * 3, 3, 1, 1)
        self.no = Normalize(nf * 3)
        self.se = SELayer(nf * 3)
        # self.no.weight.data.fill_(0)
        # self.no.bias.data.fill_(0)
        # print(torch.unique(self.no.bias.data), 'in residual init')

    def forward(self, x):
        b, t, c, h, w = x.shape
        # print(torch.unique(self.no.bias.data), 'in residual forward')
        x = x.reshape(b, t * c, h, w)
        residual = self.no(self.conv1(self.conv(x)))
        # print(torch.unique(residual))
        # residual = self.no(self.conv1(self.conv(x)))
        out = residual * self.se(residual) + x
        # chatt = self.se(residual)
        # print(torch.unique(residual), torch.unique(chatt))
        return out



class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__()
        
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        
        if permuter_config is None:
            permuter_config = {"target": "dlformer.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)
        self.residual = Residual(first_stage_config.params.embed_dim)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        #sd = torch.load(path, map_location="cpu")["state_dict"]
        #for k in sd.keys():
        #    for ik in ignore_keys:
        #        if k.startswith(ik):
        #            self.print("Deleting key {} from state_dict.".format(k))
        #            del sd[k]
        #self.load_state_dict(sd, strict=False)
        #print(f"Restored from {path}")
        self.transformer.load_state_dict(torch.load(path, map_location=self.device))
        print('load from ', path)
    def init_first_stage_from_ckpt(self, config):


        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key

        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, mask_input, sudo_mask):
     
        b, t, c, h, w = x.shape
        both_mask = torch.where((mask_input == 1) | (sudo_mask == 1), torch.ones_like(mask_input),
                                torch.zeros_like(mask_input))
        _, z_indices, mask_input = self.encode_to_z(x.reshape(b*t, -1, h, w), mask_input.reshape(b*t, -1, h, w))
        _, both_z_indices, both_mask = self.encode_to_z(x.reshape(b*t, -1, h, w), both_mask.reshape(b*t, -1, h, w))
        
        a_indices = z_indices

        target = z_indices
     
        both_z_indices = both_z_indices.reshape(b, t, -1)
        both_mask = both_mask.reshape(b, t, -1)
        logits, _ = self.transformer(both_z_indices, both_mask)
      
        return logits, target, mask_input, both_mask

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample_all(self, x, mask, steps=1, temperature=1.0, t=3, h=15, w = 27, sample=False, top_k=None,
                   callback=lambda k: None):
        # x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size  # make sure model can see conditioning
            logits, _ = self.transformer(x, mask)
            _, residual_ref = self.residual_consis(logits.reshape(x.shape[0], t, h, w, logits.shape[-1]))
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, :, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            b, r, d = probs.shape
            # print(torch.max(probs, dim=-1), 'in line 163')
            # sample from the distribution or take the most likely
            if sample:

                x = torch.multinomial(probs.reshape(b*r, d), num_samples=1)
            else:
                _, x = torch.topk(probs.reshape(b*r, d), k=1, dim=-1)
            # append to the sequence and continue
            x = x.reshape(b, r)

        return x, residual_ref







    @torch.no_grad()
    def encode_to_z(self, x, mask):
        quant_z, _, info = self.first_stage_model.encode(x * (1 - mask), mask)
        # print('in encode to z line 241', torch.unique(mask), mask.shape)
        mask = F.interpolate(mask, quant_z.shape[2:], mode='nearest')
        # print('in encode to z line 243', torch.unique(mask), mask.shape)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices, mask



    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()
        N = 8
        if lr_interface:
            x, mask, sudo_mask = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, mask, sudo_mask = self.get_xc(batch, N)
        x = x.to(device=self.device)
        mask = mask.to(device=self.device)
        sudo_mask = sudo_mask.to(device=self.device)
       
        x_sudomask = x *(1 - sudo_mask)
        b, t, c, h, w = x.shape
        quant_z, z_indices, mask = self.encode_to_z(x.reshape(b*t, c, h, w), mask.reshape(b*t, 1, h, w))
        sudo_quant_z, sudo_z_indices, sudo_mask = self.encode_to_z(x.reshape(b*t, c, h, w), sudo_mask.reshape(b*t, 1, h, w))

        
        # vi infer
        index_allinfer, resdual_ref = self.sample_all(z_indices.reshape(b, t, -1), mask.reshape(b, t, -1), top_k=top_k if top_k is not None else 100, callback=callback if callback is not None else lambda k: None)
        x_index_allinfer = self.decode_to_img(index_allinfer.reshape(b*t, -1), quant_z.shape)
        _, _, cc, hh, ww = resdual_ref.shape
        resdual_ref_img = self.first_stage_model.decode(resdual_ref.reshape(b*t, cc, hh, ww))

        # sudo vi infer
        sudo_index_allinfer, _ = self.sample_all(sudo_z_indices.reshape(b, t, -1), sudo_mask.reshape(b, t, -1), top_k=top_k if top_k is not None else 100, callback=callback if callback is not None else lambda k: None)
        sudo_x_index_allinfer = self.decode_to_img(sudo_index_allinfer.reshape(b*t, -1), quant_z.shape)

        #print('in line 198 transformer hole log image', torch.unique(index_allinfer), torch.unique(sudo_index_allinfer))

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)
        x_rec_sudo = self.decode_to_img(sudo_z_indices, quant_z.shape)
        # x_sudo = self.decode_to_img(z_indices_both, quant_z.shape)

        print(x.shape, x_sudomask.shape, x_rec.shape, x_rec_sudo.shape, x_index_allinfer.shape, sudo_x_index_allinfer.shape, 'fhuaighr')
        log["inputs"] = x.reshape(b*t, c, h, w)
        log['sudo_input'] = x_sudomask.reshape(b*t, c, h, w)
        log["reconstructions"] = x_rec
        log["reconstructions_sudo"] = x_rec_sudo
        log['vi_res'] = x_index_allinfer
        log['sudo_vi_res'] = sudo_x_index_allinfer
        log['resref'] = resdual_ref_img


        return log

    def get_input(self, key, batch):

        x = batch[key]
        #print('in get input', key, type(x), x[0].shape, type(x[0]), x.shape)
        #print('in get input', type(x), type(x[0]), x[0].shape)
        
        #x = torch.stack(x, 0)
        #print('in get input', key, type(x), x[0].shape, type(x[0]), x.shape)
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        # c = self.get_input(self.cond_stage_key, batch)
        mask = self.get_input('mask', batch)
        sudo_mask = self.get_input('sudo_mask', batch)
        uni = torch.unique(mask).shape[0]
        #print('in get input', torch.unique(mask), uni, mask.shape, sudo_mask.shape, x.shape)
        if uni > 2:
            mask = torch.where(mask > 0.05, torch.ones_like(mask), torch.zeros_like(mask))
        if uni == 2:
            mask = torch.where(mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask))
        ori_length = len(mask.shape)
        if ori_length == 5:
            b, t, _,  h, w = mask.shape
        else:
            mask = mask.unsqueeze(0)
          
            b, t, _,  h, w = mask.shape
        if N is not None:
            x = x[:, :N]
            # c = c[:N]
            mask = mask[:,:N]
            sudo_mask = sudo_mask[:, :N]
        if ori_length == 4:
            mask = mask.squeeze(0)
       
        window_size = 8
        mask = mask.reshape(b*t, 1, h, w)
        mask = mask.reshape(b*t, 1, h // window_size, window_size, w // window_size, window_size).permute(0, 1, 2, 4, 3, 5).reshape(b*t, 1, (h // window_size)*(w // window_size), window_size*window_size)
        mask_windowsum = torch.sum(mask, dim=-1, keepdim=True)
        mask = torch.where(mask_windowsum>0, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.reshape(b*t, 1, h // window_size, w // window_size, window_size, window_size).permute(0, 1, 2, 4, 3, 5).reshape(b*t, 1, h, w)
        mask = mask.reshape(b, t, 1, h, w)
        
        
        return x, mask, sudo_mask

    def shared_step(self, batch, batch_idx):
        x, mask, sudo_mask = self.get_xc(batch)
        b, t, c, h, w = x.shape
        logits, target, mask_input, both_mask = self(x, mask, sudo_mask)
        mask_input = mask_input.reshape(logits.shape[0], -1)
        both_mask = both_mask.reshape(logits.shape[0], -1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction='none')

        hole_region = torch.sum(loss * ((both_mask - mask_input).reshape(-1))) / (
                    1e-8 + torch.sum((both_mask - mask_input).reshape(-1)))

        valid_region = torch.sum(loss * ((1 - both_mask).reshape(-1))) / (1e-8 + torch.sum((1 - both_mask).reshape(-1)))

        ###########residual loss
        
        quant_z, resimg = self.residual_consis(logits.reshape(b, t, h // 16, w // 16, -1).detach())
        prev, midd, latter = resimg[:, 0], resimg[:, 1], resimg[:, 2]
        _, _, cc, hh, ww = resimg.shape
        prev_mask, midd_mask, latter_mask = mask[:, 0], mask[:, 1], mask[:, 2]
        # unimask = torch.where((prev_mask + midd_mask + latter_mask) > 0, torch.ones_like(prev_mask),
        #                       torch.zeros_like(prev_mask))
        # print(resimg.shape, prev.shape, midd.shape, latter.shape, mask.shape, 'before mse loss')
        consis_loss = F.l1_loss(prev , midd ) + F.l1_loss(midd, latter)
        detail_loss = F.l1_loss(resimg, quant_z.reshape(b, t, cc, hh, ww))
         
        return hole_region, valid_region, consis_loss, detail_loss
    
    def residual_consis(self, logits, top_k=100):
            # probs = F.softmax(logits, dim=-1)
        b, t, h, w, _ = logits.shape
        logits = self.top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        d = probs.shape[-1]
        _, x = torch.topk(probs.reshape(b * t * h * w, d), k=1, dim=-1)
        x = x.reshape(b, -1)
        index = self.permuter(x, reverse=True)
        quant_z = self.first_stage_model.quantize.get_codebook_entry(index.reshape(-1),
                                                                     shape=(b * t, h, w, self.residual.nf))
        # print(quant_z.shape, 'in line 210')
        res = self.residual(quant_z.reshape(b, t, self.residual.nf, h, w)).reshape(b, t, self.residual.nf, h, w)
        #x = self.first_stage_model.decode(res.reshape(b * t, self.residual.nf, h, w)).reshape(b, t, 3, h * 16, w * 16)
        # print('in residual consis', x.shape, res.shape, quant_z.shape)
        return quant_z, res


    def training_step(self, batch, batch_idx):
        hole_loss, valid_loss, consis_loss, detail_loss = self.shared_step(batch, batch_idx)
        loss = hole_loss + valid_loss  + detail_loss + consis_loss*0.1
      
        self.log("consis_loss/loss", consis_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("detail_loss/loss", detail_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)    
        
                   
        self.log("train_hole/loss", hole_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_valid/loss", valid_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        hole_loss, valid_loss, consis_loss, detail_loss  = self.shared_step(batch, batch_idx)
    

        self.log("consis_loss/loss", consis_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("detail_loss/loss", detail_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)   
        loss = hole_loss + valid_loss + consis_loss*0.1 + detail_loss*10
        
      
        self.log("val_hole/loss", hole_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_valid/loss", valid_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        #no_decay.add('pos_emb_tempo')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer


