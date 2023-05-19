import cv2
import numpy as np
import time
import imutils
from imutils.video import VideoStream
import RPi.GPIO as gpio
import time
import numpy as np
import serial 
import threading
import os
from datetime import datetime
import requests
import cv2
import io
import socket
import struct
import time
import pickle
import zlib
from json import loads


# Define pin allocations for Ultrasonic Sensor
trig = 16
echo = 18

# initialize the video stream and wait for the camera to warm up
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Socket Initialisation
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('0.0.0.0',1 ))
connection = client_socket.makefile('wb')
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

###################Thread Stop Flags###################
imu_stop = 'l'
camera_stop = 'l'
#######################################################

button_right = int(0)
button_left = int(0)
counterBR = np.uint64(0)
counterFL = np.uint64(0)
ser = serial.Serial('/dev/ttyUSB0', 9600)
count = 0
key = 's'
distance_covered = 0
x_axis = [0]
y_axis = [0]
current_angle = 0
delta_angle = 0
radius_threshold = 160
block_set = 0
camera_dict ={}

##############################################################################################################
########################################### Parallel Threads #################################################
##############################################################################################################


################################## Camera Thread ##########################################
def object_tracking_data():
    global radius, camera_dict, angle, vs, block_detected, camera_stop, delta_angle, block_picked, colorLower, colorUpper, email_frame ,img_counter
    # loop over the frames from the video stream
    while True:
        # grab the next frame from the video stream
        frame = vs.read()
        frame = imutils.resize(frame, width=640, height=480)
        frame = cv2.flip(frame, -1)
        try:
            '''
            Sending Frames Via Socket
            '''
            #print("Sending frame")
            result, frame1 = cv2.imencode('.jpg', frame, encode_param)
            #    data = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(frame1, 0)
            size = len(data)
            ##print("{0}: {1}".format(img_counter, size))
            client_socket.sendall(struct.pack(">L", size) + data)
            img_counter += 1
        except : 
            pass
        # if the 'q' key is pressed, stop the loop
        if camera_stop == 'q':
            # cleanup the camera and close any open windows
            cv2.destroyAllWindows()
            vs.stop()
            break


################################## IMU Thread ##########################################                        
def imu_thread():
    global current_angle
    global ser
    global imu_stop
    count = 0
    while imu_stop!='q':
        if(ser.in_waiting > 0):
            count += 1
            # Read serial stream
            line = ser.readline()
            #print(line)
            
            # Avoid first n-lines of serial information  
            if count > 10:
                # Strip serial stream of extra characters
                line = line.rstrip().lstrip()
                #print(line)
                
                line = str(line)
                line = line.strip("'")
                line = line.strip("b'")
                line = line.strip("\t")
                #print(line)
                
                # Return float
                current_angle = float(line)
                
                #print(line,"\n")
            # if imu_stop == 'q':
            #     break
#########################################################################################
#########################################################################################
#########################################################################################




#########################################################################################
################################### Distance Function ###################################
#########################################################################################
gpio.setwarnings(False)
gpio.setmode(gpio.BOARD)
gpio.setup(trig, gpio.OUT)
gpio.setup(echo, gpio.IN)


def compute_distance():
    global distance
    global distance_stop 

    # Run it in a loop
    while True:
        #global distance
        # Consider the overshooting
        rest_time = 0.4

        # Ensure output has no value
        gpio.output(trig, False)
        time.sleep(0.01)

        
        # Generate trigger pulse
        gpio.output(trig, True)
        time.sleep(0.00001)
        gpio.output(trig, False)

        # Generate ECHO time signal while gpio.input (ECHO) 0: pulse start time.time() 
        pulse_start = time.time()
        tmt = pulse_start + rest_time
        while gpio.input (echo) == 0 and pulse_start < tmt:
            pulse_start = time.time()
        
        pulse_end = time.time()
        tmt = pulse_end + rest_time
        while gpio.input (echo) == 1 and pulse_end < tmt:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start

        # Convert time to distance
        distance = pulse_duration*17150
        distance = round(distance, 2)
        distance = distance/100
        #print("Distance: ", distance)
    
#########################################################################################
################################### Motor Functions #####################################
#########################################################################################
#### Initialize GPIO pins ####
def init():
    print("k")
    gpio.setmode(gpio.BOARD)
    gpio.setup(31, gpio.OUT)
    gpio.setup(33, gpio.OUT)
    gpio.setup(35, gpio.OUT)
    gpio.setup(37, gpio.OUT)

    gpio.setup(7, gpio.IN, pull_up_down = gpio.PUD_UP) # Encoder Input Pin for Front Left Wheel
    gpio.setup(12, gpio.IN, pull_up_down = gpio.PUD_UP) # Encoder Input Pin for Back Right Wheel
#### Gameover function ####
def gameover():
    gpio.output(31, False)
    gpio.output(33, False)
    gpio.output(35, False)
    gpio.output(37, False)
    gpio.cleanup()
#### Gameover function ####
def stop():
    gpio.output(31, False)
    gpio.output(33, False)
    gpio.output(35, False)
    gpio.output(37, False)
#########################################################################################
#########################################################################################
#########################################################################################



#########################################################################################
################################### Turn Functions ######################################
#########################################################################################
        
def turn_with_imu(pwm_right, pwm_left):
    # 1 wheel revolution = 20 ticks
    # 2*pi*r distance = 1 wheel revolution
    # 20/(2*pi*r) = 1 meter
    global button_left
    global button_right
    global current_angle
    val_left = 50
    val_right = 50
    pwm_left.start(val_left)
    pwm_right.start(val_right)
    time.sleep(0.1)
    right_encoder_pulses = []
    left_encoder_pulses = []
    if int(gpio.input(12)) != int(button_right):
        button_right = int(gpio.input(12))
    if int(gpio.input(7)) != int(button_left):
        button_left = int(gpio.input(7))
    pwm_left.ChangeDutyCycle(val_left)# + (kp * delta))
    pwm_right.ChangeDutyCycle(val_right)# + (kp * delta))

def turn_with_imu_angle(angle):
    global current_angle
    val_left = 45
    val_right = 45
    target_angle = angle
    time.sleep(0.1)
    delta_angle_imu = 1000
    delta_angle_threshold = 4
    # Check what quadrant the angle is in and based on the quadrant and angle to cover to reach the target angle, decide the direction of rotation
    while np.abs(delta_angle_imu) > delta_angle_threshold:
        if target_angle > current_angle:
            if target_angle - current_angle > 180:
                delta_angle_imu = 360 - (target_angle + current_angle)
                #print("Im in 1")
                # turn left
                turn_with_imu(pwm_pin2, pwm_pin4)
                time.sleep(0.1)
                if np.abs(delta_angle_imu) < delta_angle_threshold:
                        break
                
            else:
                delta_angle_imu = target_angle - current_angle
                #print("Im in 2")
                # turn right
                turn_with_imu(pwm_pin1, pwm_pin3)
                time.sleep(0.1)
                if np.abs(delta_angle_imu) < delta_angle_threshold:
                    break
        else:
            if current_angle - target_angle > 180:
                #print("Im in 3")
                delta_angle_imu = 360 - (current_angle + target_angle)
                # turn right
                turn_with_imu(pwm_pin1, pwm_pin3)
                time.sleep(0.1)
                if np.abs(delta_angle_imu) < delta_angle_threshold:
                    break
            else:
                #print("Im in 4")
                delta_angle_imu = current_angle - target_angle
                # turn left
                turn_with_imu(pwm_pin2, pwm_pin4)
                time.sleep(0.1)
                if np.abs(delta_angle_imu) < delta_angle_threshold:
                    break
    pwm_pin1.stop()
    pwm_pin2.stop()
    pwm_pin3.stop()
    pwm_pin4.stop()
    print("Current angle: ",current_angle)


def align_robot():
    global current_angle
    global delta_angle
    print("Aligning robot")
    if delta_angle < 0:
        print("Turning right")
        turn_with_imu(pwm_pin1, pwm_pin3)
        pwm_pin1.stop()
        pwm_pin3.stop()
        button_right = int(0)
        button_left = int(0)

    elif delta_angle > 0:
        print("Turning left")
        turn_with_imu(pwm_pin2, pwm_pin4)
        pwm_pin2.stop()
        pwm_pin4.stop()
        button_right = int(0)
        button_left = int(0)
def calculate_orientation_offset(x):
    direction = ""
    degree = (x - (640/2))*0.061
    if(degree < 0):
        direction = 'left'
    else:
        direction = 'right'
    return degree,direction
#########################################################################################
#########################################################################################
#########################################################################################

#Flask Server initalisation to establish communication between Leaders and Followers
#url is in in form of the local ip address with port number as 1
url_post_status = 'http://0.0.0.0:1/follower1_status_update'
url_get_system_status = 'http://0.0.0.0:1/system_status'
system_status = None
get_system_status_stop = "i"

tracking_variables = None
depth = 0
centroid_x,centroid_y = 0,0
# Start a thread for for hitting the get_status endpoint every second and update the system_status variable
def get_system_status():
    global get_system_status_stop
    global tracking_variables
    global system_status,centroid_x,centroid_y,depth
    depth_data = tracking_variables['depth_data'] 
    depth = depth_data["depth_agg"]
    system_status = tracking_variables['system_status'] 
    tracking_data = tracking_variables['tracking_data'] 
    centroid_x,centroid_y = tracking_data['pred_data']
    
    while get_system_status_stop != 'q':
        response = requests.get(url_get_system_status)
        if response.status_code == 200:
            system_status = response.json()
            #return system_status
        else:
            system_status = "None"
            print('Error:', response.status_code)
    return 

def post_status(status):
    data = {'status': status}
    response = requests.post(url_post_status, data=data)
    print(response.txt)
    return 

#def get_tracking_information():
#    global tracking_variables,depth,centroid_x,centroid_y
#    depth = tracking_variables['depth']
#    centroid_x,centroid_y = tracking_variables['centroid_x'],tracking_variables['centroid_y']
#    
##############################################################################################################
##############################################################################################################
##############################################################################################################



##############################################################################################################
####################################### Linear Motion Functions ##############################################
##############################################################################################################
def stop_robot(pwm_right, pwm_left):
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)
    return 

def trace_leader(pwm_right, pwm_left):
    global distance,delta_angle
    val_left = 15
    val_right = 15
    offset_angle = calculate_orientation_offset(centroid_x)
    if(abs(offset_angle -current_angle) > 3 ):
        delta_angle = offset_angle -current_angle
        align_robot()
    while distance > 0.3 and depth>0.3:
        pwm_left.ChangeDutyCycle(val_left)
        pwm_right.ChangeDutyCycle(val_right)
        # print("Distance: ", dis  tance)
    stop_robot(pwm_right, pwm_left)
    return

##############################################################################################################
##############################################################################################################
##############################################################################################################
    
#A Thread to Track The Images continuously and process it from the main server
t1 = threading.Thread(target= object_tracking_data)
t1.start()
time.sleep(1)
#A thread To Get The current angle from IMU 
t2 = threading.Thread(target=imu_thread)
t2.start()
time.sleep(4)    
#Thread to compute the distance of the obstacle in front using ultrasonic sensor.
t3 =  threading.Thread(target=compute_distance)
t3.start()
time.sleep(1)
#Thread to get the system status which actually establishes communication between all leaders and followers
t4 =  threading.Thread(target=get_system_status)
t4.start()
time.sleep(1)


init() 
pwm_pin1 = gpio.PWM(31, 50)# Right Wheels #  Forward
pwm_pin2 = gpio.PWM(33, 50)# Left Wheels # Forward
pwm_pin3 = gpio.PWM(35, 50)# Right Wheels #  Forward
pwm_pin4 = gpio.PWM(37, 50)# Left Wheels # Forward

pwm_pin1.start(1)
pwm_pin4.start(1)
while True:
    if(system_status !=  None):
        if(system_status['follower1_status'] != 'deactive' and  system_status['follower2_status'] != 'deactive'): 
            print("The Follower Is Able to Track the leader, Following Status: ",system_status['follower1_status'])
            trace_leader(pwm_pin1, pwm_pin4)
        else:
            print("The Follower Is Not Able to Track the leader due to Obstacle, Following Status: ",system_status['follower1_status'])

get_system_status_stop = 'q'
t4.join()
distance_stop = 'q'
t3.join()
camera_stop = 'q' 
t1.join()
gameover()
imu_stop = 'q'
t2.join() 
