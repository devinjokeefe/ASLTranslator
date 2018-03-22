import os
from PIL import Image

for root, dirs, files in os.walk(r"C:\Users\Devin\Documents\GitHub\ASL_Data\train_data"): 
    for folder in dirs:
        fpath = "C:/Users/Devin/Documents/GitHub/ASL_Data/train_data/" + folder
        print("FPath: " + fpath)
        for root, dirs, files in os.walk(fpath):
            for frame in files:
                print(frame)
                image = Image.open(fpath +"/"+frame)
               # width, height = image.size
                #h = (height - 17) / 2
             #   size = (0, 17, width, h)
              #  image = image.crop(size)
                image = image.resize((200,200), Image.ANTIALIAS)
                F_OUT = fpath + "/" + os.path.basename(frame)
                image.save(F_OUT)

        print("Finished resizing: ", folder)
