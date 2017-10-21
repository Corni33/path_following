import numpy as np


def dist_points(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def calc_dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = np.sqrt(dx*dx + dy*dy)

    return dist, u

# curvature from 3 points
# A = hf.dist_points(self.x_path[ind], self.y_path[ind],
#                 self.x_path[ind + 1], self.y_path[ind + 1])
# B = hf.dist_points(self.x_path[ind], self.y_path[ind],
#                 self.x_path[ind + 2], self.y_path[ind + 2])
# C = hf.dist_points(self.x_path[ind + 1], self.y_path[ind + 1],
#                 self.x_path[ind + 2], self.y_path[ind + 2])
# s = 0.5 * (A + B + C)
# triangle_area = np.sqrt(s * (s - A) * (s - B) * (s - C))
#
# radius = A * B * C / (4 * triangle_area)
# kappa = 1 / radius

