import os
from keras.models import load_model
import subprocess
import glob
from PIL import Image, ImageFont, ImageDraw
import numpy as np

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
    
    answers =["BOWTIE", "ADVISE_INFLUENCE", "AFRAID", "AGAIN", "ANSWER", "APPOINTMENT", "ART_DESIGN", "BIG", "BLAME", "BOSS", "CAMP", "CANCEL_CRITICIZE", "CHAT", "CITY_COMMUNITY", "COLLECT", "COME", "COP", "COPY", "CUTE", "DATE_DESSERT", "DECREASE", "DEPRESS", "DEVELOP", "DEVIL_MISCHIEVOUS", "DISAPPOINT", "DISCUSS", "DOCTOR", "DRESS_CLOTHES", "DRINK", "EAT", "EMPHASIZE", "EXCUSE", "EXPERT", "FACE", "FED-UP_FULL", "FIFTH", "FINGERSPELL", "FIRE_BURN", "GET-TICKET", "GO", "GOLD_ns-CALIFORNIA", "GOVERNMENT", "GROUND", "GUITAR", "HAPPY", "IN", "INCLUDE_INVOLVE", "INFORM", "INJECT", "ISLAND_INTEREST", "LIVE", "LOOK", "MACHINE", "MAD", "MAN", "MARRY", "NICE_CLEAN", "PAST", "POSS", "RUN", "SAME", "SHELF_FLOOR", "SHOW", "SILLY", "STAND-UP", "TOUGH", "VACATION", "WALK", "WEEKEND", "WORK-OUT"]

    
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
        Y_Pred = np.add(Y_Pred, tempArray)

        topK = np.argsort(np.array(Y_Pred))

        holder = Image.new("RGBA", (225, 70), (1, 50, 67))
        font = ImageFont.truetype(r"C:\Users\Devin\Downloads\BebasNeue.otf", 18)
        draw = ImageDraw.Draw(image2)
   
        ansOne = answers[topK[0, -1]]
        ansTwo = answers[topK[0, -2]]
        ansThree = answers[topK[0, -3]]

        accOne = round(Y_Pred[0, topK[0, -1]] * 100 / i, 1)
        accTwo = round(Y_Pred[0, topK[0, -2]] * 100 / i, 1)
        accThree = round(Y_Pred[0, topK[0, -3]] * 100 / i, 1)

        text = "1. {0} - {1}% certain\n2. {2} - {3}% certain\n3. {4} - {5}% certain".format(ansOne, accOne, ansTwo, accTwo, ansThree, accThree)

        font = ImageFont.truetype(r"C:\Users\Devin\Downloads\BebasNeue.otf", 18)
        draw = ImageDraw.Draw(holder)
        draw.text((5, 5), text, font=font, fill=(247, 202, 24))
        image2.paste(holder, (0, 235))
        path = r'C:/Users/Devin/Documents/GitHub/ASL_Data/VidFolder/output_{}.png'.format(i)
        image2.save(path)

    index = str(np.where(Y_Pred == Y_Pred.max()))
    index = index.replace('(array([0], dtype=int64), array([', '')
    index = index.replace('], dtype=int64))', '')

    os.chdir(r"C:\Users\Devin\Documents\GitHub\ASL_Data\VidFolder")
    subprocess.call('ffmpeg -framerate 16 -loglevel quiet -i output_%01d.png summary.mp4', shell=True)
    os.startfile('C:/Users/Devin/Documents/GitHub/ASL_Data/VidFolder/summary.mp4')

    print (answers[int(index)])

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
