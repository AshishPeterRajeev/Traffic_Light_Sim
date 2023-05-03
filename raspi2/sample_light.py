import RPi.GPIO as GPIO
import time

try:
    GPIO.setmode(GPIO.BCM)

    led = {"red":23,"yellow":24,"green":25}

    for pin in led.values:
        GPIO.setup(pin,GPIO.OUT)

    while True:
        GPIO.output(led["red"],GPIO.HIGH)
        time.sleep(1)
        GPIO.output(led["red"],GPIO.LOW)
        # time.sleep(1)
        GPIO.output(led["green"],GPIO.HIGH)
        time.sleep(5)
        GPIO.output(led["green"],GPIO.LOW)
        # time.sleep(1)
        GPIO.output(led["yellow"],GPIO.HIGH)
        time.sleep(3)
        GPIO.output(led["yellow"],GPIO.LOW)
        # time.sleep(1)
        
        
except KeyboardInterrupt:
    GPIO.cleanup()