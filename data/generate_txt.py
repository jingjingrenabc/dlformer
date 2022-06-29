import os
#img_dir = '/home/jingjingren/abc_project/taming-transformers-master/data/mydata/xingxing_t'
#img_dir = '/mnt/wfs/mmchongqingwfssz/project_mm-base-vision/jingjingren/taming-transformers-master/data/mydata/xingxing_t'
#img_dir = '/home/ubuntu/Workspace/wy/dataset/DAVIS/JPEGImages/480p/hike/'
# img_dir = '/home/ubuntu/Workspace/abc/dataset/RESIDE-6K/test/hazy/'
# #img_dir = '/mnt/wfs/mmchongqingwfssz/project_mm-base-vision/data/youtube-vos/test/generated_masks/02b1a46f42'
# imgs = sorted(os.listdir(img_dir))
# f = open("/home/ubuntu/Workspace/abc/dlformer_demo_pkg/dlformer_demo/data/txt_files/haze_test_hazzzz.txt",'a')
# for i in imgs:
#     f.write(os.path.join(img_dir, i) +'\n')
#print(i)



#cod_video_dir = '/home/ubuntu/Workspace/abc/dataset/DAVIS/test/JPEGImages/camel/'
frame_dir = 'data/JPEGImages/soccerball'
mask_dir = 'data/Annotations/soccerball'
f = open("data/txt_files/soccerball.txt",'a')
f_mask = open("data/txt_files/soccerball_mask.txt",'a')

masks = sorted(os.listdir(os.path.join(mask_dir)))
frames = sorted(os.listdir(os.path.join(frame_dir)))
for fr, fr_mask in zip(frames, masks):
        
        f_mask.write(os.path.join(mask_dir, fr_mask) + '\n')
        f.write(os.path.join(frame_dir, fr) + '\n')
