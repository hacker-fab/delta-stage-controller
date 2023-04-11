# this file is used to communicate with the motor controller board for the Delta Stage

import time
import serial
import serial.tools.list_ports
import numpy as np

BAUD_RATE = 115200
camera_angle = 44.5 #degrees

# find and open the arduino nano port
arduino_port = None
ports = list(serial.tools.list_ports.comports())
for p in ports:
	print(p)
	if "CH340" in p.description:
		arduino_port = serial.Serial(p.device, BAUD_RATE)
		print("Arduino Nano found!")

if arduino_port == None:
	print("Can't find Arduino Nano!")
#	exit() this is undefined, maybe because I'm not running in the interpreter?

# Create camera theta matrix
# copied from sanga.py
camera_theta: float = (camera_angle / 180) * np.pi
R_camera: np.ndarray = np.array(
    [
        [np.cos(camera_theta), -np.sin(camera_theta), 0],
        [np.sin(camera_theta), np.cos(camera_theta), 0],
        [0, 0, 1],
    ]
)

# Transformation matrix converting delta into cartesian
x_fac: float = -1 *  np.divide(2, np.sqrt(3), )
y_fac: float = -1 
z_fac: float = np.divide(1, 3)

Tvd: np.ndarray = np.array(
    [
        [-x_fac, x_fac, 0],
        [0.5 * y_fac, 0.5 * y_fac, -y_fac],
        [z_fac, z_fac, z_fac]])

Tdv: np.ndarray = np.linalg.inv(Tvd)


# wait for commands
time.sleep(3)
arduino_port.flush()
while (1):
    user_input = input("Please enter: <x # | y # | z #>") # x/y/z steps
    steps = int(user_input[2:])
    axis = user_input[0]
    cartesian_displacement_array = np.array([
        steps if axis == "x" else 0,
        steps if axis == "y" else 0,
        steps if axis == "z" else 0
    ])

        
    # Transform into camera coordinates
    camera_displacement_array = np.dot(R_camera, cartesian_displacement_array)

    # Transform into delta coordinates
    delta_displacement_array = np.dot(Tdv, camera_displacement_array)
    
    # Turn back into string for output to arduino
    delta_array = delta_displacement_array.round().astype(int)

    output = 'mr ' + str(delta_array[0]) + ' ' + str(delta_array[1]) + ' ' + str(delta_array[2])
    print(output)
    arduino_port.write(output.encode())
    arduino_echo = arduino_port.readline()
    print(arduino_echo)