# this file is used to communicate with the motor controller board for the Delta Stage

import time
import serial
import serial.tools.list_ports

BAUD_RATE = 115200

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

# wait for commands
time.sleep(3)
arduino_port.flush()
while (1):
	user_input = input("Please enter: <mrx # | mry # | mrz #>") # x/y/z steps
	arduino_port.write(user_input.encode())
	arduino_echo = arduino_port.readline()
	print(arduino_echo)