import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(1234)


x1 = np.random.uniform(0,1,200)


plt.hist(x1,bins='auto')
plt.title('unnormalized hist')
plt.show()


