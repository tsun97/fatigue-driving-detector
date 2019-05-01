from gpiozero import Buzzer
import time

bz = Buzzer(17, active_high=False)

try:
    bz.on()
    while True:
        i = 0
except KeyboardInterrupt:
    bz.off()