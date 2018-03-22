import os
from PIL import Image, ImageChops

size = (192,192)

for root, dirs, files in os.walk(r"C:\Users\Devin\Documents\GitHub\ASL_Data\test_data"): 
    for folder in dirs:
        fpath = "C:/Users/Devin/Documents/GitHub/ASL_Data/test_data/" + folder
        print("FPath: " + fpath)
        for root, dirs, files in os.walk(fpath):
            for frame in files:
                print(frame)
                image = Image.open(fpath +"/"+frame)
                image = image.resize(size, Image.ANTIALIAS)
                F_OUT = fpath + "/" + os.path.basename(frame)
                image.save(F_OUT)
         
        print("Finished resizing: ", folder)
