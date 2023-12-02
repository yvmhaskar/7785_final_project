import math
import numpy as np
import matplotlib.pyplot as plt

# Function that takes current point and uses desired direction to output a new position and orientation
def calculateNewWaypoint(x, y, w, dir):
    if dir == 1:
        dw = -math.pi / 2.0
        local_p1 = np.array([0, 1, 0, 1])
    elif dir == 2:
        dw = math.pi / 2.0
        local_p1 = np.array([0, -1, 0, 1])
    elif dir == 3:
        dw = math.pi
        local_p1 = np.array([-1, 0, 0, 1])
    else:
        dw = 0
        local_p1 = np.array([0, 0, 0, 1])

    # Calculate Pnew once outside the loop
    T = np.array([
        [math.cos(w), -math.sin(w), 0, x],
        [math.sin(w), math.cos(w), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Pnew = np.dot(T, local_p1)
    Wnew = w+dw
    Wnew = np.arctan2(np.sin(Wnew), np.cos(Wnew))
    Pnew[2] = Wnew
    x_new = Pnew[0]
    y_new = Pnew[1]
    w_new = Pnew[2]
    return x_new, y_new, w_new, dir

# Number of iterations
num_iterations = 5

# Initial position and orientation
x_cur = 1.0
y_cur = 1.0
w_cur = 0.0

# Expand the plot window
plt.figure(figsize=(10, 10))

# Plotting
plt.scatter(x_cur, y_cur, marker='o', color='blue', label='Initial Position')
scale_factor = 0.02*plt.gcf().get_size_inches()[0]
for i in range(1, num_iterations+1):
    dir = np.random.choice([1, 2])
    x_cur, y_cur, w_cur, dur = calculateNewWaypoint(x_cur, y_cur, w_cur, dir)
    print(x_cur, y_cur, w_cur,dir)
    plt.scatter(x_cur, y_cur, marker='o', color='red')
    #plt.quiver(x_cur, y_cur, np.cos(w_cur), np.sin(w_cur), angles='xy', scale_units='xy', scale=scale_factor, color='green')
    plt.text(x_cur, y_cur, str(i), fontsize=10, ha = 'right', va = 'bottom')

# Final position and orientation
plt.scatter(x_cur, y_cur, marker='o', color='orange', label='Final Position')
#plt.quiver(x_cur, y_cur, np.cos(w_cur), np.sin(w_cur), angles='xy', scale_units='xy', scale=scale_factor, color='green', label='Final Direction')
plt.text(x_cur, y_cur, 'End', fontsize=10, ha = 'right', va = 'bottom')

plt.axis('equal')
plt.axis('auto')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Movement')
plt.legend()
plt.grid(True)
plt.show()
