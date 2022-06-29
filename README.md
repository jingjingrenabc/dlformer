# Dlformer
This is implementation of CVPR 2022 paper DLFormer Discrete Latent Transformer for Video Inpainting

## Dependency
```
conda env create -f environment.yaml
```
## Inference
Set model.params.ckpt_path as path of the model（download [here](https://drive.google.com/file/d/1bOiInDYYEZyfxDRgtueGPH6r9h1QaGsd/view?usp=sharing)）and model.params.first_stage_config.ckpt_path as path of the model (download [here](https://drive.google.com/file/d/1DdsOR0fyxR0V_vGR7-cPrd8cuq2swoKj/view?usp=sharing))
```
python test_transformer_sppe.py -c configs/test_breakdance_transformer.yaml -s save_dir
```
## Training your own model
1) Fine tune the autoencoder together with a codebook on current video
```
python train_vqgan.py --base configs/train_breakdance_vqgan.yaml  -t True --gpus 0,
```
2) Set ckpt_path in train_breakdance_vqgan.yaml as the path of checkpoint produced in 1) , set mode as select, and select the codes used in the current video by running
```
python test_vqgan.py -c configs/train_breakdance_vqgan.yaml -s save_dir
```
3) In train_breakdance_transformer.yaml set model.params.first_stage_config.ckpt_path as the path of model produced in 2), model.params.first_stage_config.n_embed as the code selected in 2), model.params.transformer_config.params.vocab_size as the code selected in 2), train the transformer for code inference by running
```
python train_transformer.py --base configs/train_breakdance_transformer.yaml  -t True --gpus 0,
```
4) Set model.params.ckpt_path in train_breakdance_transformer.yaml as the transformer path obtained in 3) and get the result by running
```
python test_transformer_sppe.py -c configs/train_breakdance_transformer.yaml -s save_dir
```
## Citation
```
@inproceedings{ren2022dlformer,
  title={DLFormer: Discrete Latent Transformer for Video Inpainting},
  author={Ren, Jingjing and Zheng, Qingqing and Zhao, Yuanyuan and Xu, Xuemiao and Li, Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3511--3520},
  year={2022}
}
```
## Acknowledgement
Our code is based on [VQGAN](https://github.com/CompVis/taming-transformers) and [STTN](https://github.com/researchmm/STTN). Thanks for their code sharing.

