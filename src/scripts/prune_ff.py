import os
from shutil import copyfile
import random
random.seed(42)
datapath="data/faceforensics/test_set"
destination = "data/faceforensics/test_set_pruned"
CUTOFF = 15
for real_or_fake in os.listdir(datapath):
    for imgfolder in os.listdir(os.path.join(datapath, real_or_fake)):
        t = 0
        ls = os.listdir(os.path.join(datapath, real_or_fake, imgfolder))
        random.shuffle(ls)
        cutoff = min(CUTOFF, len(ls))
        while(t<cutoff):
            src = os.path.join(datapath, real_or_fake, imgfolder, ls[t])
            dst = os.path.join(destination, real_or_fake, imgfolder, ls[t])
            if not os.path.exists(os.path.join(destination, real_or_fake, imgfolder)):
                os.makedirs(os.path.join(destination, real_or_fake, imgfolder))
            copyfile(src, dst)
            t += 1

