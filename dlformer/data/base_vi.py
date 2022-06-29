import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import cv2
import torch
from dlformer.data.generate_mask import create_random_shape_with_random_motion
from torchvision import transforms 
import random
class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, paths_mask, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["mask_path_"] = paths_mask
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler_img = albumentations.SmallestMaxSize(max_size = self.size)
            #self.rescaler_img = albumentations.Compose([rescaler_img])
            self.rescaler_mask = albumentations.SmallestMaxSize(max_size = self.size, interpolation=cv2.INTER_NEAREST)
            #self.rescaler_mask = albumentations.Compose([rescaler_mask])
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.cropper], additional_targets={"mask": "image"})
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, mask_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        #print(image.shape, mask.shape, type(mask), np.unique(mask), 'in base data perpocess image')
        image = self.rescaler_img(image=image)['image']
        mask = self.rescaler_mask(image=mask)['image']
        #print(image.shape, mask.shape, type(mask), np.unique(mask),'in base data perpocess image')
        dataprocess = self.preprocessor(image=image, mask=mask)
        image, mask = dataprocess['image'], dataprocess['mask']
        
        image = (image/127.5 - 1.0).astype(np.float32)
        mask = mask / 255
        return image, mask

    def __getitem__(self, i):
        example = dict()
        #print(self._length, len(self.labels["file_path_"]), len(self.labels['mask_path_']))
        example["image"], example['mask'] = self.preprocess_image(self.labels["file_path_"][i], self.labels['mask_path_'][i])
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        for k in self.labels: #return additional information of the sample e.g. img path, mask path..
            example[k] = self.labels[k][i]
        return example


class ImagePaths_trans(Dataset):
    def __init__(self, paths, paths_mask, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["mask_path_"] = paths_mask
        self._length = len(paths)
        if self.size is not None and self.size > 0:
            self.rescaler_img = albumentations.SmallestMaxSize(max_size = self.size)
            self.rescaler_mask = albumentations.SmallestMaxSize(max_size = self.size, interpolation=cv2.INTER_NEAREST) 
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([ self.cropper], additional_targets={"mask": "image", "coord":"image"})
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, mask_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        image = self.rescaler_img(image=image)['image']
        mask = self.rescaler_mask(image=mask)['image']
        h, w, _ = image.shape
        #print(image.shape, 'in line 98 in Imagepath base vi ')
        coord = np.arange(h * w).reshape(h, w, 1) / (h * w)
        d2 = self.preprocessor(image=image, mask=mask, coord=coord)
        image, mask, coord = d2['image'], d2['mask'], d2['coord']
        image = (image/127.5 - 1.0).astype(np.float32)
        mask = mask / 255
        return image, mask, coord

    def __getitem__(self, i):
        example = dict()
        example["image"], example['mask'], example['coord'] = self.preprocess_image(self.labels["file_path_"][i], self.labels['mask_path_'][i])
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        for k in self.labels: #return additional information of the sample e.g. img path, mask path..
            example[k] = self.labels[k][i]
        #example['mask'] = example['mask'].unsqueeze(-1)
        example['mask'] = np.expand_dims(example['mask'], axis=-1)
        #print('in base vi ', example["image"].shape, example['mask'].shape, example['coord'].shape)
        #in base vi  (256, 256, 3) (256, 256, 1) (256, 256, 1)
        return example


class ImagePaths_trans_sudomask(Dataset):
    def __init__(self, paths, paths_mask, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["mask_path_"] = paths_mask
        self._length = len(paths)
        self.totensor = transforms.ToTensor()
        if self.size is not None and self.size > 0:
            self.rescaler_img = albumentations.SmallestMaxSize(max_size = self.size)
            self.rescaler_mask = albumentations.SmallestMaxSize(max_size = self.size, interpolation=cv2.INTER_NEAREST)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([ self.cropper], additional_targets={"mask": "image", "coord":"image"})
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, mask_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        image = self.rescaler_img(image=image)['image']
        mask = self.rescaler_mask(image=mask)['image']
        h, w, _ = image.shape

        #print(image.shape, 'in line 98 in Imagepath base vi ')
        coord = np.arange(h * w).reshape(h, w, 1) / (h * w)
        d2 = self.preprocessor(image=image, mask=mask, coord=coord)
        image, mask, coord = d2['image'], d2['mask'], d2['coord']
        image = (image/127.5 - 1.0).astype(np.float32)
        mask = mask / 255
       
        sudo_mask = create_random_shape_with_random_motion(1, imageHeight=image.shape[0], imageWidth=image.shape[1])
        
        sudo_mask = self.totensor(sudo_mask[0].convert('L'))
        
        return image, mask, coord, sudo_mask

    def __getitem__(self, i):
        example = dict()
        example["image"], example['mask'], example['coord'], example['sudo_mask'] = self.preprocess_image(self.labels["file_path_"][i], self.labels['mask_path_'][i])
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        for k in self.labels: #return additional information of the sample e.g. img path, mask path..
            example[k] = self.labels[k][i]
        #example['mask'] = example['mask'].unsqueeze(-1)
        example['mask'] = np.expand_dims(example['mask'], axis=-1)
        #print('in base vi ', example["image"].shape, example['mask'].shape, example['coord'].shape)
        #in base vi  (256, 256, 3) (256, 256, 1) (256, 256, 1)
        return example




class ImagePaths_trans_sudomask_orishape(Dataset):
    def __init__(self, paths, paths_mask, h=None, w=None, random_crop=False, labels=None):
        self.h = h
        self.w = w
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["mask_path_"] = paths_mask
        self._length = len(paths)
        self.totensor = transforms.ToTensor()
        if self.h is not None and h > 0:
            self.rescaler_img = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)
            self.rescaler_mask = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, mask_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        image = self.rescaler_img(image=image)['image']
        mask = self.rescaler_mask(image=mask)['image']
        h, w, _ = image.shape

        #print(image.shape, np.unique(mask), 'in line 98 in Imagepath base vi preprocess image')
        image = (image / 127.5 - 1.0).astype(np.float32)
        mask = mask / 255

        sudo_mask = create_random_shape_with_random_motion(1, imageHeight=image.shape[0], imageWidth=image.shape[1])

        sudo_mask = self.totensor(sudo_mask[0].convert('L'))

        return image, mask, sudo_mask
    def __getitem__(self, i):
        example = dict() 
        example["image"], example['mask'], example['sudo_mask'] = self.preprocess_image(self.labels["file_path_"][i], self.labels['mask_path_'][i])
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        for k in self.labels: #return additional information of the sample e.g. img path, mask path..
            example[k] = self.labels[k][i]
        #example['mask'] = example['mask'].unsqueeze(-1)
        example['mask'] = np.expand_dims(example['mask'], axis=-1)
        #print('in base vi ', example["image"].shape, example['mask'].shape, np.unique(example['mask']))
        #in base vi  (256, 256, 3) (256, 256, 1) (256, 256, 1)
        example['frame_id'] = i
        return example




class ImagePaths_trans_sudomask_hazetest(Dataset):
    def __init__(self, paths, h=None, w=None, random_crop=False, labels=None):
        self.h = h
        self.w = w
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.totensor = transforms.ToTensor()
        if self.h is not None and h > 0:
            self.rescaler_img = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.rescaler_img(image=image)['image']
        h, w, _ = image.shape
        #print(image.shape, np.unique(mask), 'in line 98 in Imagepath base vi preprocess image')
        image = (image / 127.5 - 1.0).astype(np.float32)


        return image
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        example['frame_id'] = i
        return example




class ImagePaths_trans_vcod(Dataset):
    def __init__(self, paths, is_mask, h=None, w=None, random_crop=False, labels=None):
        self.h = h
        self.w = w
        self.random_crop = random_crop
        self.is_mask = is_mask
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.totensor = transforms.ToTensor()
        if self.h is not None and h > 0:
            self.rescaler_img = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)
            self.rescaler_mask = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        if not self.is_mask:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
        else:
            image = Image.open(image_path).convert('L')
        image = np.array(image).astype(np.uint8)
        orishape = (image.shape[0], image.shape[1])
        # print(type(image), image.shape, 'in dataset line 314')
        if self.is_mask:
            image = self.rescaler_mask(image=image)['image']
            h, w = image.shape
            image = np.expand_dims(image, axis=-1)
            image = image/255
        else:
            image = self.rescaler_img(image=image)['image']
            h, w, _ = image.shape
            image = (image / 127.5 - 1.0).astype(np.float32)
        #print(image.shape, np.unique(mask), 'in line 98 in Imagepath base vi preprocess image')

        return image, orishape
    def __getitem__(self, i):
        example = dict()
        example["image"], example['orishape'] = self.preprocess_image(self.labels["file_path_"][i])
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        example['video'] = self.labels["file_path_"][i].split('/')[-3]
        example['frame_id'] = i

        return example





class ImagePaths_trans_vcod_videotrans(Dataset):
    def __init__(self, img_paths, mask_paths, tl=2,  h=None, w=None, random_crop=False, labels=None):
        self.h = h
        self.w = w
        self.random_crop = random_crop
        self.labels = dict() if labels is None else labels
        self.tl = tl
        self.labels["mask_path_"] = mask_paths
        self.labels["img_path_"] = img_paths
        self._length = len(mask_paths)
        self.totensor = transforms.ToTensor()
        if self.h is not None and h > 0:
            self.rescaler = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)

        self.video_rec =dict()
        for idx, img in enumerate(self.labels['img_path_']):
            video = img.split('/')[-3]
            if self.video_rec.__contains__(video):
                self.video_rec[video].append(idx)
            else:
                self.video_rec[video] = []
                self.video_rec[video].append(idx)
       
    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, is_mask):
        if not is_mask:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
        else:
            image = Image.open(image_path).convert('L')
        image = np.array(image).astype(np.uint8)
        orishape = (image.shape[0], image.shape[1])
        # print(type(image), image.shape, 'in dataset line 314')
        image = self.rescaler(image=image)['image']
        if is_mask:
            image = np.expand_dims(image, axis=-1)
            image = image/255
        else:
            image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, orishape
    def __getitem__singleframe(self, i):
        example = dict()
        example["image"], example['orishape'] = self.preprocess_image(self.labels["img_path_"][i], is_mask = False)
        example["mask"], _ = self.preprocess_image(self.labels['mask_path_'][i], is_mask=True)
        example['name'] = self.labels["img_path_"][i].split('/')[-1]
        example['video'] = self.labels["img_path_"][i].split('/')[-3]
        example['frame_id'] = i

        return example

    def __getitem__(self, i):
        example = dict()
        example["image"] = []
        example['orishape'] = []
        example["mask"] = []
        example['name'] = []
        example['video'] = []
        example['frame_id'] = []

        video = self.labels["img_path_"][i].split('/')[-3]
        v_start, v_end = self.video_rec[video][0], self.video_rec[video][-1]

        idxs = torch.Tensor(range(i - self.tl, i + 1 + self.tl)).long().clamp(v_start, v_end)
        idxs = list(idxs)
        # idxs += [0, self._length // 2, self._length - 1]

        for idx in idxs:
            single_data = self.__getitem__singleframe(idx)
            for k, v in single_data.items():
                example[k] .append(v)

        example['image'] = torch.stack(example["image"], dim=0)
        example['mask'] = torch.stack(example["mask"], dim=0)
        example['frame_id'] = torch.Tensor(idxs) - v_start
        return example




class ImagePaths_youtube_orishape(Dataset):
    def __init__(self, paths,  h=None, w=None, random_crop=False, labels=None):
        self.h = h
        self.w = w
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        self.totensor = transforms.ToTensor()
        if self.h is not None and h > 0:
            self.rescaler_img = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)
            

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
       
        image = np.array(image).astype(np.uint8)
        image = self.rescaler_img(image=image)['image']
        h, w, _ = image.shape

        #print(image.shape, np.unique(mask), 'in line 98 in Imagepath base vi preprocess image')
        image = (image / 127.5 - 1.0).astype(np.float32)
       

        return image
    def __getitem__(self, i):
        example = dict()
        try:
            example["image"] = self.preprocess_image(self.labels["file_path_"][i])
            example['name'] = self.labels["file_path_"][i].split('/')[-1]
            for k in self.labels: #return additional information of the sample e.g. img path, mask path..
                example[k] = self.labels[k][i]
        except:
            print('load ', self.labels["file_path_"][i] , 'error')
            example = self.__getitem__( random.randint(0, self._length-1))
        #example['mask'] = example['mask'].unsqueeze(-1)
      
        #print('in base vi ', example["image"].shape, example['mask'].shape, np.unique(example['mask']))
        #in base vi  (256, 256, 3) (256, 256, 1) (256, 256, 1)
        return example



class ImagePaths_trans_sudomask_orishape_tempo(Dataset):
    def __init__(self, paths, paths_mask, h=None, w=None, tl=2, random_crop=False, labels=None):
        self.h = h
        self.w = w
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["mask_path_"] = paths_mask
        self._length = len(paths)
        self.totensor = transforms.ToTensor()
        self.temp_len = tl
        if self.h is not None and h > 0:
            self.rescaler_img = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR)
            self.rescaler_mask = albumentations.Resize(height=h, width=w, interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, mask_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        image = self.rescaler_img(image=image)['image']
        mask = self.rescaler_mask(image=mask)['image']
        h, w, _ = image.shape

        #print(image.shape, np.unique(mask), 'in line 98 in Imagepath base vi preprocess image')
        image = (image / 127.5 - 1.0).astype(np.float32)
        mask = mask / 255

        sudo_mask = create_random_shape_with_random_motion(1, imageHeight=image.shape[0], imageWidth=image.shape[1])

        sudo_mask = self.totensor(sudo_mask[0].convert('L'))
        image = self.totensor(image)
        mask = self.totensor(mask)
        return image, mask, sudo_mask
    def __getitem__(self, i):
        example = dict()
        example["image"] = []
        example['mask'] = []
        example['sudo_mask'] = []
        
        idxs = torch.Tensor(range( i - self.temp_len,  i + 1 + self.temp_len)).long().clamp(0, self._length-1)
        idxs = list(idxs)
        #idxs += [0, self._length // 2, self._length - 1]
       
        for idx in idxs:
            
            image, mask, sudo_mask = self.preprocess_image(self.labels["file_path_"][idx], self.labels['mask_path_'][idx])
            
                #image, mask, _ = self.preprocess_image(self.labels["file_path_"][idx], self.labels['mask_path_'][idx])
                #sudo_mask = torch.zeros_like(mask).float()
            #print(image.shape, sudo_mask.shape, mask.shape, type(image), type(mask), type(sudo_mask), 'in base vi get item')
            example["image"].append(image)
            example['mask'].append(mask)
            example['sudo_mask'].append(sudo_mask)
        example['image'] = torch.stack(example["image"], dim=0)
        example['mask'] = torch.stack(example["mask"], dim=0)
        example['sudo_mask'] = torch.stack(example["sudo_mask"], dim=0)
        example['frame_id'] = i
        example['tempo_ids'] = torch.Tensor(idxs)
        example['name'] = self.labels["file_path_"][i].split('/')[-1]
        for k in self.labels:
            example[k] = self.labels[k][i]

        return example



class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

