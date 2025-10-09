# let's see how timer is made in python
import time
second = int(input("Enter the time in seconds:"))
while second:
    mins, secs = divmod(second, 60)
    timer = '{:02d}:{:02d}'.format(mins, secs)
    print(timer, end="\r")
    time.sleep(1)
    second -= 1
print("Time's up!")
