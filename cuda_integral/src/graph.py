import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.system("pwd")
fig = plt.figure()

# fig.savefig('test.png')

results = []
counter = 0
with open("../cmake-build-debug/tmp/python_time") as time:
    for line in time.readlines():
        results.extend([counter for i in range(int(int(line)/100000))])
        counter += 1
plt.hist(results, bins=counter)
fig.show()
