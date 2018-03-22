import os
from keras.models import load_model
import numpy as np
import subprocess
import glob
from PIL import Image, ImageFont, ImageDraw
import pydot, graphviz
from keras.utils import plot_model

Prod = True

if (Prod):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Net = load_model(r'C:\Users\Devin\ASL_Model.h5')

def VidToImg (Video, NewFile):
    duration = subprocess.check_output(['ffprobe', '-i', Video, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")])

    keepChars = set('.0123456789')
    duration = ''.join(filter(keepChars.__contains__, str(duration)))

    end_dur = float(duration) - 4.0
    subprocess.call('ffmpeg -ss 00:00:02 -t ' + str(end_dur) + ' -loglevel quiet -i ' + Video + ' ' + NewFile, shell=True)

    duration = float(duration) - 4.0
    fps = 50.0 / duration
    os.chdir(r"C:\Users\Devin\Documents\GitHub\ASL_Data\TempFolder")
    subprocess.call('ffmpeg -i ' + NewFile + ' -loglevel quiet -vf fps=' + str(fps) + ' output_%03d.png', shell=True)

    os.remove(NewFile)

    PredictVal ()

def PredictVal ():
    
    path = 'C:/Users/Devin/Documents/GitHub/ASL_Data/TempFolder/*.png'
    Frames = glob.glob (path)
    Y_Pred = np.zeros((1, 70))
    i=0
    for Img in Frames:
        i += 1
        image = Image.open(Img)
        image = image.crop((17, 17, 225, 320))
        image2 = image
        image = image.resize((200, 200), Image.ANTIALIAS)
        imagearray = []
        imagearray = np.array(image)
        imagearray = imagearray.reshape((1, 200, 200, 3))

        tempArray = Net.predict (imagearray, steps=1)
#        layer_name = 'block14_sepconv2'
        
        plot_model(Net, to_file='model.png')
    UserInput()
    
def UserInput():
    
    text = input("Press enter to translate a video")
    if (text == ""):
        text = input("Type the filepath for your next video\n")
        text = text.replace("\\", '/')
        filename, file_extension = os.path.splitext(text)
        nfHolder = filename + "_New" + file_extension
        VidToImg(text, nfHolder)
UserInput()

def Visualise ():
    Net.summary()
  
