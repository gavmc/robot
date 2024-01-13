import cv2
import os
import numpy as np


image_path = 'D:/robot/screenshots/screenshots'
save_path = 'D:/robot/screenshots/32RGB'

sizex = 58
sizey = 32


def process(img_path, s_path, out=False):
    image = cv2.imread(img_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (sizex, sizey))
    
    if(out):
        image[image > 0] = 255
    cv2.imwrite(s_path, image)



filenames = next(os.walk(image_path+r'/raw'), (None, None, []))[2]
i=0
for f in filenames:
    process(image_path+'/raw/'+f, save_path+'/raw/'+f)
    i+=1
    if(i%1000 == 0):
        print(i)


filenames = next(os.walk(image_path+r'/checkpoint'), (None, None, []))[2]
i=0
for f in filenames:
    process(image_path+'/checkpoint/'+f, save_path+'/checkpoint/'+f, True)
    i+=1
    if(i%1000 == 0):
        print(i)

filenames = next(os.walk(image_path+r'/gate'), (None, None, []))[2]
i=0
for f in filenames:
    process(image_path+'/gate/'+f, save_path+'/gate/'+f, True)
    i+=1
    if(i%1000 == 0):
        print(i)

filenames = next(os.walk(image_path+r'/ring'), (None, None, []))[2]
i=0
for f in filenames:
    process(image_path+'/ring/'+f, save_path+'/ring/'+f, True)
    i+=1
    if(i%1000 == 0):
        print(i)

filenames = next(os.walk(image_path+r'/wall'), (None, None, []))[2]
i=0
for f in filenames:
    process(image_path+'/wall/'+f, save_path+'/wall/'+f, True)
    i+=1
    if(i%1000 == 0):
        print(i)

    
