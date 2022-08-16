from distutils import cmd
from multiprocessing import cpu_count
from turtle import width
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import ffmpeg
import numpy as np

class Video_ffmpeg_Loader(Dataset):

    def __init__(self, dataset, sequence_length, scale,crop_width,crop_height,mood):
        #Dataset is dictionary object include video path
        self.dataset=dataset
    

        #trimm
        self.sequence_length=sequence_length

        #resize
        self.scale=scale

        #crop
        self.crop_width=crop_width
        self.crop_height=crop_height

        if mood=='train':
            self.crop_fn=self.crop_train
        else:
            self.crop_fn=self.crop_val
    
    def __getitem__(self, index):
        YOUR_FILE= self.dataset[index]

        #read video information
        probe = ffmpeg.probe(YOUR_FILE)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)


        height=int(video_stream['height'])
        width=int(video_stream['width'])
        number_frame=int(video_stream['nb_frames'])
        time = float(video_stream['duration']) 
      
       #compute coff or resize then crop the images
        if width>height:
            new_height=str(self.scale) 
            coff=height/self.scale
            new_width=str(int(width//coff))
            x,y=self.crop_fn(new_width,new_height)
   
        else:
            new_width=str(self.scale)
            coff=width/self.scale
            new_height=str(int(height//coff))
            x,y=self.crop_fn(new_width,new_height)

            
        bit_rate=number_frame/time

        extra_frame=0
        if self.sequence_length > number_frame:
            sequence_l=number_frame
            extra_frame=self.sequence_length-number_frame
        else:
            sequence_l=self.sequence_length
        
        #randomly select number frame to trim
        starting_point=np.random.randint(number_frame-sequence_l+1, size=1)
        
        ss_i=starting_point[0]/bit_rate
        t_i=sequence_l/bit_rate        

        
        #applying trimm, resize and crop by ffmpeg command
        out, _ =(
            ffmpeg
            .input(YOUR_FILE ,ss=ss_i,t=t_i).setpts('PTS-STARTPTS')
            .filter('scale',new_width,new_height)
            .filter('crop', w=self.crop_width, h=self.crop_height , x=str(x), y=str(y))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(quiet=True)
        )

        #to_numpy 
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, int(self.crop_width), int(self.crop_height), 3])
        )

        if extra_frame:
            video=np.pad(video, [(0, extra_frame), (0, 0),(0,0),(0,0)], mode='constant', constant_values=0)

        return video
        
    
            
    def __len__(self):
        return self.dataset.__len__()

    def crop_train(self,new_width,new_height):
        x=np.random.randint((int(new_width)-self.crop_width)/2, size=1)
        y=np.random.randint((int(new_height)-self.crop_height)/2, size=1)
        return x[0],y[0]


    def crop_val(self,new_width,new_height):
        x=int((int(new_width)-self.crop_width)/2)
        y=int((int(new_height)-self.crop_height)/2)
        return x,y


