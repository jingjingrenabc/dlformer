import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from utils import  DataModuleFromConfig
from dlformer.utils.utils import instantiate_from_config
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange
import torch.nn.functional as F
# python test_vqgan.py --resume ./logs/2021-09-10T11-07-55_youtube_vqgan/checkpoints/last.ckpt --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.custom_vi_trans_sudomask.CustomTest_specific}}}" --outdir ./result_car_turn2


def save_image(x, path):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)


@torch.no_grad()
def run_conditional(model, dsets, outdir, top_k, temperature, savedir, batch_size=1):
    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))
    codeset = set([])
    print("Dataset: ", dset.__class__.__name__)
    for start_idx in trange(0,len(dset)-batch_size+1,batch_size):
        indices = list(range(start_idx, start_idx+batch_size))
        example = default_collate([dset[i] for i in indices])

        x, mask = model.get_input( example, 'image', 'mask')
        x, mask = x.to(model.device), mask.to(model.device)
        for i in range(x.shape[0]):
            save_image(x[i], os.path.join(outdir, "originals",
                                          "{:06}.png".format(indices[i])))

        # cond_key = model.cond_stage_key
        # c = model.get_input(cond_key, example).to(model.device)
        # print('input x shape', x.shape )
        # scale_factor = 1.0
        quant_z, _, (_, _, z_indices) = model.encode(x*(1-mask), mask)
        codeset = codeset|set([int(i) for i in list(torch.unique(z_indices))])
        mask = F.interpolate(mask, quant_z.shape[2:], mode='nearest')

        #mask = F.interpolate(mask, quant_z.shape[2:], mode='nearest')
        #print(len(torch.unique(z_indices)), torch.unique(mask.reshape(405)*z_indices), mask.shape, x.shape, z_indices.shape)
        #quant_c, c_indices = model.encode_to_c(c)
        
        cshape = quant_z.shape

        #print(len(torch.unique(z_indices)), len(z_indices), z_indices.shape, cshape, '$$$$$$$$$$$$$$$$$$$$')
        xrec = model.decode(quant_z)
        #xrec = model.decode(quant_z)
        for i in range(xrec.shape[0]):
            save_image(xrec[i], os.path.join(outdir, "reconstructions",
                                             "{:06}.png".format(indices[i])))
           
        # if start_idx == 5:
        #     candidate = torch.unique(z_indices)
        #     for i in candidate:
        #         ind = torch.ones_like(z_indices)*i
        #         xrec_ind = model.decode_code(ind, (cshape[0], cshape[2], cshape[3], cshape[1]))
        #         for g in range(xrec_ind.shape[0]):
        #             save_image(xrec_ind[g], os.path.join(outdir, "single",
        #                                              "{:06}.png".format(int(i))))


    print(len(codeset), 'codes selected')
    idx_used = torch.Tensor(sorted(list(codeset))).long().to(model.device)
    model.quantize.embedding.weight = torch.nn.Parameter(model.quantize.embedding(idx_used))
    model.n_embed  = len(idx_used)
    torch.save(model.state_dict(), os.path.join(savedir, 'last_select.ckpt'))
def get_parser():
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    '''
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    
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
    '''
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass
    '''
    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
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
    '''
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base
    ''' 
    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    #if opt.ignore_base_data:
    #    for config in configs:
    #        if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)


    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    #print(f"Global step: {global_step}")

    outdir = os.path.join(opt.save_dir, 'images', 'test')
    
    os.makedirs(outdir, exist_ok=True)
    weight_dir = os.path.join(opt.save_dir, 'checkpoints')
    print("Writing samples to ", outdir)
    for k in ["originals", "reconstructions", "single"]:
        os.makedirs(os.path.join(outdir, k), exist_ok=True)
    run_conditional(model, dsets, outdir, opt.top_k, opt.temperature, weight_dir)
    #print(weight.shape,'end of training')
    #torch.save(weight, os.path.join(weight_dir, 'weight_used.ckpt'))
