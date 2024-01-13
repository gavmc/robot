import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import RPi.GPIO as GPIO  #only works on linux
from time import sleep

official_map = np.zeros((9, 9))

for i in range(9):
    official_map[i][0] = 1
    official_map[i][8] = 1
    official_map[0][i] = 1
    official_map[8][i] = 1

for i in range(3):
    for j in range(3):
        official_map[(i*2)+2][(j*2)+2] = 1


in_pins = [4, 27, 22, 5, 6, 13, 26, 23]
en_pins = [24, 25, 12, 17]
power = []
'''
in1 = 4
in2 = 27
in3 = 22
in4 = 5
in5 = 6
in6 = 13
in7 = 26
in8 = 23

en1 = 24
en2 = 25
en3 = 12
en4 = 17
'''
#in1-4 / en1-2 = left side
#in5-8 / en3-4 = right side

def startup():
    global power
    GPIO.setmode(GPIO.BCM)
    for i in range(len(in_pins)):  
        GPIO.setup(in_pins[i], GPIO.OUT)

    for i in range(len(en_pins)):  
        GPIO.setup(en_pins[i], GPIO.OUT)

        power.append(GPIO.PWM(en,1000))
        power[:-1].start(25) # sets initial speed 0-100



def move_motor(motor, direction):  # motor 0-3, direction 1=forward 0=stop -1=backwards
    if(direction == 0):
        GPIO.output(in_pins[(motor*2)], GPIO.LOW)
        GPIO.output(in_pins[(motor*2)+1], GPIO.LOW)
        
    if(direction == 1):
        GPIO.output(in_pins[(motor*2)], GPIO.HIGH)
        GPIO.output(in_pins[(motor*2)+1], GPIO.LOW)
        
    if(direction == -1):
        GPIO.output(in_pins[(motor*2)], GPIO.LOW)
        GPIO.output(in_pins[(motor*2)+1], GPIO.HIGH)


def forward():
    for i in range(4):
        move_motor(i, 1)

def backward():
    for i in range(4):
        move_motor(i, -1)


def left():   # might have to rework these depending on the wiring
    for i in range(2):
        move_motor(i, -1)
    for i in range(2, 4):
        move_motor(i, 1)

def right():  # might have to rework these depending on wiring
    for i in range(2):
        move_motor(i, 1)
    for i in range(2, 4):
        move_motor(i, -1)

def stop():
    for i in range(4):
        move_motor(i, 0)



image = cv2.imread(r'D:\robot\screenshots\32RGB\raw\7.png')/255


gate_predictor = tf.keras.models.load_model('D:/robot/gate_ml')
ring_predictor = tf.keras.models.load_model('D:/robot/ring_ml')
wall_predictor = tf.keras.models.load_model('D:/robot/wall_ml')

def make_prediction(img):
    img = np.array(img).reshape((1, 32, 58, 3))
    wall_img = wall_predictor(img)[0]
    gate_img = gate_predictor(img)[0]
    ring_img = ring_predictor(img)[0]
    img = np.array(img).reshape((32, 58, 3))


    wall_img = tf.where(wall_img < .3, 0, 1)
    gate_img = tf.where(gate_img < .75, 0, 1)
    ring_img = tf.where(ring_img < .8, 0, 1)

    color_diff = np.sqrt(((img - np.array([.3, .3, 1])) ** 2).sum(axis=-1))
    threshold = .25  # Adjust this threshold as needed
    close_pixels = np.where(color_diff < threshold)

    checkpoint = np.zeros((32, 58))
    for i in range(len(close_pixels[0])):
        checkpoint[close_pixels[0][i]][close_pixels[1][i]] = 1

    robot = np.zeros((32, 58))
    for i in range(32):
        for j in range(58):
            if(25 <= i <= 31 and 13 <= j <= 18):
                robot[j][i] = 1

    return (np.array(wall_img), np.array(gate_img), np.array(ring_img), np.array(checkpoint), np.array(robot))


#ring - 1
#wall - 2
#gate - 3
#checkpoint - 4
#robot - 5

def get_map(data):
    wall_img, gate_img, ring_img, checkpoint, robot = data
    complete_map = np.zeros((9, 9))
    
    for i in range(9):
        complete_map[i][0] = 1
        complete_map[i][8] = 1
        complete_map[0][i] = 1
        complete_map[8][i] = 1

    for i in range(3):
        for j in range(3):
            complete_map[(i*2)+2][(j*2)+2] = 1

    t_ring = np.array(ring_img).reshape((32, 58))

    x_min = 100
    x_max = -1

    y_min = 100
    y_max = -1

    for i in range(len(t_ring)):
        for j in range(len(t_ring[0])):
            if(t_ring[i][j] == 1):
                if(i < y_min):
                    y_min = i
                if(i > y_max):
                    y_max = i

                if(j < x_min):
                    x_min = j
                if(j > x_max):
                    x_max = j

    gate_img_t = np.array(gate_img)[y_min:y_max+1, x_min:x_max+1]
    wall_img_t = np.array(wall_img)[y_min:y_max+1, x_min:x_max+1]
    ring_img_t = np.array(ring_img)[y_min:y_max+1, x_min:x_max+1]
    robot_t = np.array(robot)[y_min:y_max+1, x_min:x_max+1]
    checkpoint_t = np.array(checkpoint)[y_min:y_max+1, x_min:x_max+1]


    segment_dist = None
    segment_dim = -1

    if(x_min != 0 and x_max != 57):
        segment_dist = (x_max - x_min) / 4
        segment_dim = 1

    elif(y_min != 0 and y_max != 31):
        segment_dist = (y_max - y_min) / 4
        segment_dim = 2

    if(segment_dist != None):
        if(segment_dim == 1):
            if(y_min != 0):
                for j in range(int(np.floor((y_max-y_min)/segment_dist))):
                    for i in range(4):
                        if(wall_img_t[int(round(segment_dist*(j+1)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))+1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))-1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2


                        if(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2



                        direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                        hit = 0
                        for k in range(4):
                            c_hit = 0
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))] == 1):
                                    hit += 1
                                    c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))+direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))+direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))-direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))-direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                           
                        if(hit > 2):
                            complete_map[(j*2)+1][(i*2)+1] = 3

                        if(robot_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+1][(i*2)+1] = 5

                if(np.any(checkpoint_t == 1)):
                    for j in range(int(np.floor((y_max-y_min)/segment_dist))):
                        for i in range(4):
                            if(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))+1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))-1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))+1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))-1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
            else:
                for j in range(9-int(np.floor((y_max-y_min)/segment_dist)), 9):
                    for i in range(4):
                        if(wall_img_t[int(round(segment_dist*(j+1)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))+1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))-1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2


                        if(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2



                        direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                        hit = 0
                        for k in range(4):
                            c_hit = 0
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))] == 1):
                                    hit += 1
                                    c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))+direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))+direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))-direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))-direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                           
                        if(hit > 2):
                            complete_map[(j*2)+1][(i*2)+1] = 3

                        if(robot_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+1][(i*2)+1] = 5

                if(np.any(checkpoint_t == 1)):
                    for j in range(int(np.floor((y_max-y_min)/segment_dist))):
                        for i in range(4):
                            if(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))+1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))-1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))+1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))-1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4

        else:
            if(x_min != 0):
                for j in range(4):
                    for i in range(int(np.floor((x_max-x_min)/segment_dist))):
                        if(wall_img_t[int(round(segment_dist*(j+1)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))+1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))-1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2


                        if(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2



                        direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                        hit = 0
                        for k in range(4):
                            c_hit = 0
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))] == 1):
                                    hit += 1
                                    c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))+direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))+direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))-direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))-direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                           
                        if(hit > 2):
                            complete_map[(j*2)+1][(i*2)+1] = 3

                        if(robot_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+1][(i*2)+1] = 5

                if(np.any(checkpoint_t == 1)):
                    for j in range(int(np.floor((x_max-x_min)/segment_dist))):
                        for i in range(4):
                            if(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))+1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))-1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))+1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))-1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4




            else:
                for j in range(4):
                    for i in range(9-int(np.floor((x_max-x_min)/segment_dist)), 9):
                        if(wall_img_t[int(round(segment_dist*(j+1)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))+1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2
                        elif(wall_img_t[int(round((segment_dist*(j+1))-1))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+2][(i*2)+1] = 2


                        if(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2
                        elif(wall_img_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round(segment_dist*(i+1)))] == 1):
                            complete_map[(j*2)+1][(i*2)+2] = 2



                        direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                        hit = 0
                        for k in range(4):
                            c_hit = 0
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))] == 1):
                                    hit += 1
                                    c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))+direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))+direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                            try:
                                if(gate_img_t[int(round((segment_dist*j)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][0]))-direction[k][0]][int(round((segment_dist*i)+(segment_dist/2))) + int(round(segment_dist/2*direction[k][1]))-direction[k][1]] == 1):
                                    if(c_hit == 0):
                                        hit += 1
                                        c_hit = 1
                            except:
                                pass
                           
                        if(hit > 2):
                            complete_map[(j*2)+1][(i*2)+1] = 3

                        if(robot_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                            complete_map[(j*2)+1][(i*2)+1] = 5

                if(np.any(checkpoint_t == 1)):
                    for j in range(int(np.floor((x_max-x_min)/segment_dist))):
                        for i in range(4):
                            if(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))+1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))-1][int(round((segment_dist*i)+(segment_dist/2)))] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))+1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
                            elif(checkpoint_t[int(round((segment_dist*j)+(segment_dist/2)))][int(round((segment_dist*i)+(segment_dist/2)))-1] == 1):
                                complete_map[(j*2)+1][(i*2)+1] = 4
    else:
        return None
    return complete_map

def explore():
    wall, _, ring, _, _ = make_prediction(image)

    obstacle = np.zeros((len(wall), len(wall[0])))
    for i in range(len(wall)):
        for j in range(len(wall[0])):
            if(wall[i][j] == 1):
                obstacle[i][j] == 1
            elif(ring[i][j] == 1):
                obstacle[i][j] == 1
    #28, 16
    direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    max_dist = 0
    max_dir = []
    for k in range(4):
        dist = 0
        while True:
            try:
                if(obstacle[28+(direction[k][0]*dist)][16+(direction[k][1]*dist)] == 1):
                    print(does_not_exist)
                dist += 1
            except:
                if(dist > max_dist):
                    max_dist = dist
                    max_dir = direction[k]

        if(max_dir == [1, 0]):
            right()
            forward()
            time.sleep(1)
            stop()
            

        elif(max_dir == [-1, 0]):
            left()
            forward()
            time.sleep(1)
            stop()
            

        elif(max_dir == [0, 1]):
            right()
            right()
            forward()
            time.sleep(1)
            stop()

        elif(max_dir == [0, -1]):
            forward()
            time.sleep(1)
            stop()


    temp_map = get_map(make_prediction(image))
    if(temp_map == None):
        explore()
    else:
        return temp_map



print(get_map(make_prediction(image)))
    

plt.imshow(image)
plt.title("Original Image")
plt.show()


