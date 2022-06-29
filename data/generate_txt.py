import os
# to generate txt file list of the frames and masks of the video
frame_dir = 'data/JPEGImages/soccerball'
mask_dir = 'data/Annotations/soccerball'
f = open("data/txt_files/soccerball.txt",'a')
f_mask = open("data/txt_files/soccerball_mask.txt",'a')

masks = sorted(os.listdir(os.path.join(mask_dir)))
frames = sorted(os.listdir(os.path.join(frame_dir)))
for fr, fr_mask in zip(frames, masks):
        
        f_mask.write(os.path.join(mask_dir, fr_mask) + '\n')
        f.write(os.path.join(frame_dir, fr) + '\n')
