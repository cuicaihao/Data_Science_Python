import numpy as np


def compute_sides(coord):

    ax = coord[0, 0]
    ay = coord[1, 0]  # Coordinates of the point A
    bx = coord[0, 1]
    by = coord[1, 1]  # Coordinates of the point B
    cx = coord[0, 2]
    cy = coord[1, 2]  # Coordinates of the point C

    ab = compute_length(ax, ay, bx, by)
    bc = compute_length(bx, by, cx, cy)
    ca = compute_length(cx, cy, ax, ay)
    return ab, bc, ca


def compute_length(p1x, p1y, p2x, p2y):
    return np.sqrt(np.power(p1x-p2x, 2)+np.power(p1y-p2y, 2))


def compute_area(coord):
    """ make sure the triangle exists"""
    [a, b, c] = sorted(compute_sides(coord))
    if a + b - c < np.finfo(float).eps:
        return 0
    s = (a + b + c) / 2
    return (s*(s-a)*(s-b)*(s-c)) ** 0.5


def circum_inside(coord):
    ax = coord[0, 0]
    ay = coord[1, 0]  # Coordinates of the point A
    bx = coord[0, 1]
    by = coord[1, 1]  # Coordinates of the point B
    cx = coord[0, 2]
    cy = coord[1, 2]  # Coordinates of the point C

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    # Coordinates of the circumcenter
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by)
          * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by)
          * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d

    # Radius of the circle
    r = np.sqrt((ax - ux)**2 + (ay - uy)**2)

    # Determine whether or not the circumcircle is covered by the unit circle
    area = compute_area(coord)
    if np.sqrt(ux**2+uy**2) + r <= 1 and area > np.finfo(float).eps:
        return True


def is_inside(a):
    return a[0]**2 + a[1]**2 <= 1


N = 10**6  # Number of Monte Carlo samples in both methods

# Generate samples by rejecting/accepting

count = 0  # Initialize the number of desired groups
k = 0
data_inside = np.zeros((2, 3))

while k <= 3*N:
    pt = np.random.uniform(-1, 1, 2)  # draw one point
    if is_inside(pt):  # Determine if the point lies inside the unit circle
        s = k % 3
        data_inside[0, s] = pt[0]
        data_inside[1, s] = pt[1]
        if k % 3 == 2 and circum_inside(data_inside):
            count += 1
        k += 1

per_rej = count / N

# Generate samples by polar coordinates
ct = 0  # Initialize the number of desired groups
for i in range(N):
    data_polar = np.array([np.sqrt(np.random.random(3)),
                           np.random.uniform(-np.pi, np.pi, 3)])
    # Tranform polar coornates to cartesian coordinates
    sp = data_polar.copy()
    sc = np.array([sp[0, :]*np.cos(sp[1, ]), sp[0, :]*np.sin(sp[1, ])])
    if circum_inside(sc):
        ct += 1

per_polar = ct / N

print("The Monte Carlo method (N={}) tell the answer is:".format(N))
print("polar     coordinates: {:.6f}".format(per_polar))
print("Cartesian coordinates: {:.6}".format(per_rej))
print("But formula 2*pi/15  : {:.6}".format(2*np.pi/15))
