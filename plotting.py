import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


x = np.random.randint(0,20,size = 20)
y = np.random.randint(0,20,size = 20)
z = np.random.randint(0,20,size = 20)
size = np.random.randint(0,10,size = 20)
arr = np.random.randint(0,20,size = (20, 3))
df = pd.DataFrame(arr)

df.hist()
plt.show()

df.plot.box()
plt.show()

plt.scatter(x,y,s = size * 200)
plt.show()

plt.bar(x,y)
plt.show()

plt.plot(x,y)
plt.show()

plt.pie(x)
plt.show()

ax = plt.axes(projection = '3d')
ax.plot3D(x,y,z)
