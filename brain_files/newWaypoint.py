import math

# givens
pi = math.pi
x_cur = 1.0
y_cur = 1.0
w_cur = pi / 2.0
dir = 1  # 1 right, 2 left, 3 back, 0 stay

# function that takes current point and uses desired direction to output a new position and orientation
def newWaypt(x, y, w, dir):
    waypoints = [[1, 1], [1, 2], [2, 1], [2, 2]]

    if dir == 1:
        dw = -pi / 2.0
        local_p1 = [0, 1, 0, 1]
    elif dir == 2:
        dw = pi / 2.0
        local_p1 = [0, -1, 0, 1]
    elif dir == 3:
        dw = pi
        local_p1 = [-1, 0, 0, 1]

    # Calculate Pnew once outside the loop
    T = [
        [math.cos(dw), -math.sin(dw), 0, x],
        [math.sin(dw), math.cos(dw), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    Pnew = [sum(T[i][j] * local_p1[j] for j in range(4)) for i in range(4)]

    errors = []
    for waypoint in waypoints:
        x_target, y_target = waypoint

        # Calculate error for each waypoint
        error = math.sqrt((x_target - Pnew[0]) ** 2 + (y_target - Pnew[1]) ** 2)
        errors.append(error)

    return errors

# Example usage
errors = newWaypt(x_cur, y_cur, w_cur, dir)
print(f'Errors from each waypoint: {errors}')
