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

    return Pnew

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

for _ in range(num_iterations):
    dir = np.random.choice([1, 2, 3, 0])
    x_cur, y_cur, w_cur = calculateNewWaypoint(x_cur, y_cur, w_cur, dir)
    plt.scatter(x_cur, y_cur, marker='o', color='red')
    plt.quiver(x_cur, y_cur, np.cos(w_cur), np.sin(w_cur), angles='xy', scale_units='xy', scale=0.5, color='green')

# Final position and orientation
plt.scatter(x_cur, y_cur, marker='o', color='orange', label='Final Position')
plt.quiver(x_cur, y_cur, np.cos(w_cur), np.sin(w_cur), angles='xy', scale_units='xy', scale=0.5, color='green', label='Final Direction')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Movement')
plt.legend()
plt.grid(True)
plt.show()
