import math
import numpy as np

# function that takes current point and uses desired direction to output a new position and orientation
def newWaypt(x, y, w, dir):
    waypoints = np.array[[-0.25,0.0],[0.6,0.0],[0.6,-0.9],[-0.25,-0.9],[-0.25,-1.8],[-1.1,-1.8],[-1.1,-0.9],[-1.1,0.0],[-2.0,0.0],[-2.0,-0.9],[-2.0,-1.8],[-2.9,-1.8],[-2.9,-0.9],[-2.9,0.0],[-3.8,0.0],[-3.8,-0.9],[-3.8,-1.8]]

    if dir == 1:
        dw = -pi / 2.0
        local_p1 = np.array([0, 1, 0, 1])
    elif dir == 2:
        dw = pi / 2.0
        local_p1 = np.array([0, -1, 0, 1])
    elif dir == 3:
        dw = pi
        local_p1 = np.array([-1, 0, 0, 1])

    # Calculate Pnew once outside the loop
    T = np.array([
        [math.cos(w), -math.sin(w), 0, x],
        [math.sin(w), math.cos(w), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Pnew = np.dot(T, local_p1)
    min_error = float('inf')
    newWaypt = None

    for waypoint in waypoints:
        x_target, y_target,w_temp = waypoint

        # Calculate error for each waypoint
        error = math.sqrt((x_target - Pnew[0]) ** 2 + (y_target - Pnew[1]) ** 2)
        if error < min_error:
            min_error = error
            newWaypoint = waypoint

    newWaypoint[2] = w+dw

    Wnew = w+dw
    Wnew = np.arctan2(np.sin(Wnew), np.cos(Wnew))
    Pnew[2] = Wnew
    
    return Pnew


# givens
pi = math.pi
x_cur = 1.0
y_cur = 1.0
w_cur = 0.0
dir = 2  # 1 right, 2 left, 3 back, 0 stay

print(x_cur, y_cur, w_cur)
# Example usage
Pnew = newWaypt(x_cur, y_cur, w_cur, dir)
print(f'New waypoint: {Pnew}')
x_new = Pnew[0]
y_new = Pnew[1]
w_new = Pnew[2]
print(f'New waypoint: {newWaypt(x_new, y_new, w_new, 3)}')
