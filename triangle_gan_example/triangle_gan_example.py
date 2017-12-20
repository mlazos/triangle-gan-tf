import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(0, 1.0, 1000)
y = np.random.normal(0, 1.0, 1000)


print(x)
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
