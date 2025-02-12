import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

d1 = 0.1
d2 = 0.05

#discretization with N = 250, dx = 1
#we start with the homogeneous and stationary solution (u, v) = (1, 0)
uv = np.zeros((2, 250, 250))
uv[0, :, :] = 1 #because u = 1

#we perturb this by changing the values on a (20, 20) square where (u, v) = (0.5, 0.5) 
#plus an additive noise of 0.1 that value
u_new = 0.5 * (1 + 0.1 * np.random.randn())
v_new = 0.5 * (1 + 0.1 * np.random.randn())
zero_start = np.random.randint(0, 230) #we do this so we have an index where to start and place the 20x20 square on
one_start = np.random.randint(0, 230)
print(zero_start)
uv[0, zero_start:zero_start+20, one_start:one_start+20] = u_new
uv[1, zero_start:zero_start+20, one_start:one_start+20] = v_new
#print(uv[0, zero_start-2:zero_start+2, one_start-2:one_start+2])