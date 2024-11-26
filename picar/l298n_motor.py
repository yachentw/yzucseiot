#!/usr/bin/python
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#|R|a|s|p|b|e|r|r|y|P|i|.|c|o|m|.|t|w|
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Copyright (c) 2016, raspberrypi.com.tw
# All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# l298n_motor.py
# control a dc motor clockwise and counterclockwise
#
# Author : sosorry
# Date   : 08/01/2015

import RPi.GPIO as GPIO
import time
import sys

Motor_Pin1 = 16
Motor_Pin2 = 18

if len(sys.argv) > 1:
    if sys.argv[1] == "left":
        Motor_Pin1 = 11
        Motor_Pin2 = 13
    else: # right or others
        Motor_Pin1 = 16
        Motor_Pin2 = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(Motor_Pin1, GPIO.OUT)
GPIO.setup(Motor_Pin2, GPIO.OUT)

try:
    GPIO.output(Motor_Pin1, True)     # clockwise
    time.sleep(3)
    GPIO.output(Motor_Pin1, False)

    time.sleep(1)                       # protect motor

    GPIO.output(Motor_Pin2, True)     # counterclockwise
    time.sleep(3)
    GPIO.output(Motor_Pin2, False)

finally:
    GPIO.cleanup()

