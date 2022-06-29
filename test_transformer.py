import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
import os
from PIL import Image
from dlformer.utils.utils import instantiate_from_config

from torch.utils.data.dataloader import default_collate


rescale = lambda x: (x + 1.) / 2.
normalize = lambda x: (x - x.min()) / (x.max() - x.min())

def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def save_img(xstart, fname):
    I = (xstart.clip(0,1)[0]*255).astype(np.uint8)
    Image.fromarray(I).save(fname)



def get_interactive_image(resize=False):
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"])
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        print("upload image shape: {}".format(image.shape))
        img = Image.fromarray(image)
        if resize:
            img = img.resize((256, 256))
        image = np.array(img)
        return image


def single_image_to_torch(x, permute=True):
    assert x is not None, "Please provide an image through the upload function"
    x = np.array(x)
    x = torch.FloatTensor(x/255.*2. - 1.)[None,...]
    if permute:
        x = x.permute(0, 3, 1, 2)
    return x


def pad_to_M(x, M):
    hp = math.ceil(x.shape[2]/M)*M-x.shape[2]
    wp = math.ceil(x.shape[3]/M)*M-x.shape[3]
    x = torch.nn.functional.pad(x, (0,wp,0,hp,0,0,0,0))
    return x

@torch.no_grad()
def run_conditional(model, dsets, save_dir):
    
    dsets = dsets.datasets['validation']
    recs= []
    vi_res = []
   
    for idx,  data in enumerate(dsets):
               
        curr = 0 if idx == 0 else 1
        x, mask, _,  = model.get_xc(data) 
       
        x, mask = x.unsqueeze(0), mask.unsqueeze(0)
        b, t, _, h, w = x.shape
        x, mask = x.to(model.device), mask.to(model.device)
        _, z_indices, mask = model.encode_to_z(x.reshape(b*t, 3, h, w), mask.reshape(b*t, 1, h, w))
        _, index_allinfer = model.sample_all(z_indices.reshape(b, t, -1), mask.reshape(b, t, -1))
        x_index_allinfer = model.first_stage_model.decode(index_allinfer.squeeze(0))
       
        x_index_allinfer = x_index_allinfer[curr].unsqueeze(0)
        Image.fromarray((rescale((x_index_allinfer.cpu().squeeze(0).permute(1, 2, 0)).clamp(-1, 1).numpy() )* 255).astype(np.uint8)).save(
            os.path.join(save_dir, 'vi' + str(idx) + '.jpg'))

        print('processing', idx)


    return recs, vi_res



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save_dir",

        help="Ignore data specification from base configs. Useful if you want "
             "to specify a custom datasets on the command line.",
        type=str
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )

    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):


    model = instantiate_from_config(config)


    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data



def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    #print('in load model and dset', ckpt, config)
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step





if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    ckpt = None


    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)



    gpu = True
    eval_mode = True
    show_config = False

    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    os.mkdir(opt.save_dir)
    run_conditional(model, dsets, opt.save_dir)
    print('finished')



