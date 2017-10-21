import numpy as np
from scipy.interpolate import splprep, splev
import helper_functions as hf


class Simulator:
    def __init__(self):
        self.state = None
        self.dt = None
        self.L = None
        self.sim_solution = None
        self.sim_time_vector = None
        self.look_ahead_point = None
        self.x_path = None
        self.y_path = None
        self.kappa_path = None
        self.theta_path = None

    def init(self, path_spline, num_points=100, L=3.0, dt=0.1):
        self.L = L
        self.dt = dt

        self.x_path, self.y_path = splev(np.linspace(0, 1, num_points), path_spline)
        xp, yp = splev(np.linspace(0, 1, num_points), path_spline, der=1)
        xpp, ypp = splev(np.linspace(0, 1, num_points), path_spline, der=2)

        self.kappa_path = []
        for xp_,xpp_,yp_,ypp_ in zip(xp,xpp,yp,ypp):
            self.kappa_path.append((xp_*ypp_ - xpp_*yp_)/(xp_**2+yp_**2)**1.5) # curvature at each point

        self.theta_path = []
        for i in range(len(self.x_path)-1):
            dx = self.x_path[i + 1] - self.x_path[i]
            dy = self.y_path[i + 1] - self.y_path[i]
            self.theta_path.append(np.arctan(dy/dx))
        self.theta_path.append(self.theta_path[-1])


    def reset(self, initial_state):
        self.state = initial_state
        self.sim_solution = [initial_state]
        self.sim_time_vector = [0.0]
        self.look_ahead_point = []

    def unroll_state(self):
        return [s for s in self.state]

    def move(self, delta, a=0):
        x, y, theta, v = self.unroll_state()

        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += v/self.L * np.tan(delta) * self.dt # tan(..) is left out in udacity
        v += a * self.dt

        self.state = [x, y, theta, v]

        self.sim_solution.append(self.state)
        self.sim_time_vector.append(self.sim_time_vector[-1] + self.dt)

    def finalize(self):
        self.sim_solution = np.array(self.sim_solution)
        self.sim_time_vector = np.array(self.sim_time_vector)
        self.look_ahead_point = np.array(self.look_ahead_point)

    def look_ahead(self, look_ahead_dist):
        #find closest point on target path
        min_dist = 9999999
        u = None
        ind = None
        for i in range(len(self.x_path)-2): #-2 because of curvature
            dist_, u_ = hf.calc_dist(self.x_path[i], self.y_path[i],
                             self.x_path[i+1], self.y_path[i+1],
                             self.state[0], self.state[1])
            if dist_ < min_dist:
                min_dist = dist_
                ind = i
                u = u_

        # walk along the path to look ahead
        dist_path_points = hf.dist_points(self.x_path[ind], self.y_path[ind],
                                      self.x_path[ind + 1], self.y_path[ind + 1])
        walk_dist = (1.0 - u) * dist_path_points

        while walk_dist < look_ahead_dist and ind < len(self.x_path)-3:
            ind += 1
            dist_path_points = hf.dist_points(self.x_path[ind], self.y_path[ind],
                                           self.x_path[ind + 1], self.y_path[ind + 1])
            walk_dist += dist_path_points

        # interpolate
        u = 1.0 - (walk_dist - look_ahead_dist)/dist_path_points

        # lookahead point (interpolated)
        x_look = self.x_path[ind] + u * (self.x_path[ind + 1] - self.x_path[ind])
        y_look = self.y_path[ind] + u * (self.y_path[ind + 1] - self.y_path[ind])

        # theta (interpolated)
        theta = self.theta_path[ind] + u * (self.theta_path[ind+1] - self.theta_path[ind])

        # curvature (interpolated)
        kappa = self.kappa_path[ind] + u * (self.kappa_path[ind+1] - self.kappa_path[ind])

        self.look_ahead_point.append([x_look, y_look, theta, kappa])

        return (x_look, y_look, theta, kappa)







        #return (x_near, y_near, theta, kappa)





