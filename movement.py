import RPi.GPIO as GPIO  #only works on linux
from time import sleep


# probably gonna have to move pins around to make sure motors are connected to the right input
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
    for i in range(len(in_pins)):  # sets input pins
        GPIO.setup(in_pins[i], GPIO.OUT)

    for i in range(len(en_pins)):  # sets enabler pins 
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

