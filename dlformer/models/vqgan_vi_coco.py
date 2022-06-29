import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from dlformer.utils.utils import instantiate_from_config,  get_obj_from_str
# from pytorch_lightning.utilities.distributed import rank_zero_only
from dlformer.modules.diffusionmodules.model import Encoder, Decoder
from dlformer.modules.vqvae.quantize import VectorQuantizer2_vi as VectorQuantizer
from dlformer.modules.vqvae.quantize import GumbelQuantize


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 mask_key='mask',
                 mode='train', 
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
       
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        if ckpt_path is not None:
            if mode == 'train':
                self.init_from_coco(ckpt_path, ignore_keys=ignore_keys)
            else:
                if mode == 'select':
                    self.load_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
                else:
                    if mode == 'eval':
                        self.load_from_ckpt_forinfer(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        self.mask_key = mask_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        


    def init_from_coco(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        encoder_dict = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder')}
        self.encoder.load_state_dict(encoder_dict)

        decoder_dict = {k.replace('decoder.', ''): v for k, v in sd.items() if k.startswith('decoder')}
        self.decoder.load_state_dict(decoder_dict)

        quantize_dict = {k.replace('quantize.', ''): v for k, v in sd.items() if k.startswith('quantize')}
        self.quantize.load_state_dict(quantize_dict)

        before_dict = {k.replace('quant_conv.', ''): v for k, v in sd.items() if k.startswith('quant_conv')}
        self.quant_conv.load_state_dict(before_dict)

        post_dict = {k.replace('post_quant_conv.', ''): v for k, v in sd.items() if k.startswith('post_quant_conv')}
        self.post_quant_conv.load_state_dict(post_dict)

        print('load models ckpt from coco prtrained')

    def load_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def load_from_ckpt_forinfer(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def encode(self, x, mask):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # print('in line 61 encode to z', h.shape, mask.shape)
        quant, emb_loss, info = self.quantize(h, mask)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, mask):
        quant, diff, info = self.encode(input, mask)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k, mask):
        x, mask, sudo_mask = batch[k], batch[mask], batch['sudo_mask']

        if len(x.shape) == 3:
            x = x[..., None]
        # print(x.shape, mask.shape, torch.unique(mask), torch.max(x), torch.min(x), type(mask), type(x), 'in vqgan vi')
        uni = torch.unique(mask).shape[0]
        if uni > 2:
            mask = torch.where(mask > 0.05, torch.ones_like(mask), torch.zeros_like(mask))
        if uni == 2:
            mask = torch.where(mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask))
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        mask = mask.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        window_size = 8
        b, _, h, w = mask.shape
        # print('in get input vqgan', mask.shape)
        mask = mask.reshape(b, 1, h // window_size, window_size, w // window_size, window_size).permute(0, 1, 2, 4, 3,
                                                                                                        5).reshape(b, 1,
                                                                                                                   (
                                                                                                                               h // window_size) * (
                                                                                                                               w // window_size),
                                                                                                                   window_size * window_size)
        mask_windowsum = torch.sum(mask, dim=-1, keepdim=True)
        mask = torch.where(mask_windowsum > 0, torch.ones_like(mask), torch.zeros_like(mask))
        mask = mask.reshape(b, 1, h // window_size, w // window_size, window_size, window_size).permute(0, 1, 2, 4, 3,
                                                                                                        5).reshape(b, 1,
                                                                                                                   h, w)
        # print('in vqgan get input, ', x.shape, mask.shape, torch.unique(mask), torch.max(x), torch.min(x), type(mask), type(x), 'in vqgan vi')
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # mask = mask.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float(), mask.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        x, mask = self.get_input(batch, self.image_key, self.mask_key)
        xrec, qloss = self(x * (1 - mask), mask)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, mask, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, mask, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x, mask = self.get_input(batch, self.image_key, self.mask_key)
        xrec, qloss = self(x * (1 - mask), mask)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, mask, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, mask, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

       
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        # print('in configure optim', self.decoder.requires_grad)
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x, mask = self.get_input(batch, self.image_key, self.mask_key)
       
        x, mask = x.to(self.device), mask.to(self.device)
        
        xrec, _ = self(x * (1 - mask), mask)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x * (1 - mask)
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


