import sys
import numpy as np


source_list = []
for i in range(1, len(sys.argv)):
    source_list.append(sys.argv[i])

print(source_list)

final = np.zeros((12500,125))

for i in source_list:
    get = np.load(i)
    final += get

final = final / (len(sys.argv)-1)

target = open('Third.txt', 'w')
for i in final:
    target.write(str(i.argmax() + 1) + '\n')  # 输出

target.close()