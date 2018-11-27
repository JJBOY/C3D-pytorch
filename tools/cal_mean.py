import os
from PIL import Image
import numpy as np
path='../raw/data/'

if __name__ == '__main__':
    sum=np.zeros((128,171,3))
    count=0
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            #print(os.path.join(fpathe,f))
            img=Image.open(os.path.join(fpathe,f))
            img=np.array(img.resize((171,128),Image.ANTIALIAS))
            sum+=img
            count+=1
        print(count)
    print(sum)
    sum=sum/count
    np.save('mean.npy',sum)