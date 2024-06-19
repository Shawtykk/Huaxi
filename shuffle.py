import random
import numpy as np
 
x = np.arange(205)
random.shuffle(x)  
# print(x)
with open("Swin-Unet/data_zEI/train.txt",'w') as f:
    for i in x:
       f.write('\n'+ str(i))