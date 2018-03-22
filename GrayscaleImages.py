import os
from PIL import Image, ImageChops

size = (300,300)

for root, dirs, files in os.walk(r"C:\Users\Devin\Documents\GitHub\ASL_Data\train_data"): 
    for folder in dirs:
        fpath = "C:/Users/Devin/Documents/GitHub/ASL_Data/train_data/" + folder
        print("FPath: " + fpath)
        for root, dirs, files in os.walk(fpath):
            for frame in files:
                print(frame)
                image = Image.open(fpath +"/"+frame)
                
                image.convert('LA')
                
                F_OUT = fpath + "/" + os.path.basename(frame)
                print ("Outfile path: " + F_OUT)
                image.save(F_OUT)
         
        print("Finished resizing: ", folder)
