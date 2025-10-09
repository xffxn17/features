# let's see how run time is calculated in python
import time

from click import clear

start_time = time.time()
# do some work
for i in range(0, 10000):
    print(i)

end_time = time.time()

run_time = end_time - start_time
print(f"Run time: {run_time} seconds")
