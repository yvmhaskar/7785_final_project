import numpy as np
import matplotlib.pyplot as plt

# Function that takes current point and uses desired direction to output a new position and orientation
def newWaypt(x, y, w, dir):
    waypoints = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    # Randomly select a new direction
    dir = np.random.choice([1, 2, 3, 0])

    if dir == 1:
        dw = -np.pi / 2.0
        local_p1 = np.array([0, 1, 0, 1])
    elif dir == 2:
        dw = np.pi / 2.0
        local_p1 = np.array([0, -1, 0, 1])
    elif dir == 3:
        dw = np.pi
        local_p1 = np.array([-1, 0, 0, 1])
    else:
        # Default case: no rotation
        dw = 0
        local_p1 = np.array([0, 0, 0, 1])

    # Calculate Pnew using NumPy
    T = np.array([
        [np.cos(dw), -np.sin(dw), 0, x],
        [np.sin(dw), np.cos(dw), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Pnew = np.dot(T, local_p1)

    # Initialize variables to keep track of the minimum error and corresponding waypoint
    min_error = float('inf')
    min_error_waypoint = None

    for waypoint in waypoints:
        x_target, y_target = waypoint

        # Calculate error for each waypoint
        error = np.sqrt((x_target - Pnew[0]) ** 2 + (y_target - Pnew[1]) ** 2)

        # Update minimum error and corresponding waypoint
        if error < min_error:
            min_error = error
            min_error_waypoint = np.copy(waypoint)  # Create a copy to avoid modifying the original array
            min_error_waypoint[2] = w  # Set the third element to the value of w

    return min_error_waypoint[0], min_error_waypoint[1], min_error_waypoint[2], dir

# Number of iterations
num_iterations = 10

# Initial position and orientation
x_cur = 1.0
y_cur = 1.0
w_cur = np.pi / 2.0
dir = 1  # Initial direction

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(x_cur, y_cur, marker='o', color='blue', label='Initial Position')

for _ in range(num_iterations):
    x_cur, y_cur, w_cur, dir = newWaypt(x_cur, y_cur, w_cur, dir)
    plt.scatter(x_cur, y_cur, marker='o', color='red')
    plt.quiver(x_cur, y_cur, np.cos(w_cur), np.sin(w_cur), angles='xy', scale_units='xy', scale=0.5, color='green')

# Final position and orientation
plt.scatter(x_cur, y_cur, marker='o', color='orange', label='Final Position')
plt.quiver(x_cur, y_cur, np.cos(w_cur), np.sin(w_cur), angles='xy', scale_units='xy', scale=0.5, color='green', label='Final Direction')

# Waypoints
waypoints = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
plt.scatter(waypoints[:, 0], waypoints[:, 1], marker='x', color='black', label='Waypoints')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Movement')
plt.legend()
plt.grid(True)
plt.show()
