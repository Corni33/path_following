import numpy as np
import helper_functions as hf

# Implementation of the flatness based path following controller from:
# 'THE ARGO AUTONOMOUS VEHICLE'S VISION AND CONTROL SYSTEMS' (1999)
#  by Massimo Bertozzi et. al.

class Controller:
    def __init__(self):
        self.L = None
        self.sim_dt = None

    def init(self, sim_dt=0.1, L=3.0):
        self.L = L
        self.sim_dt = sim_dt


    def get_steering_angle(self, current_state, current_steering_angle, look_ahead_point, eta):

        (x_B, y_B, theta_B, kappa_B) = look_ahead_point
        (x_A, y_A, theta_A, v) = current_state
        (eta_1, eta_2, eta_3, eta_4) = eta

        # current curvature
        kappa_A = (1.0/self.L)*np.tan(current_steering_angle)

        # polynomial coefficients
        x0 = x_A
        x1 = eta_1 * np.cos(theta_A)
        x2 = 0.5 * (eta_3 * np.cos(theta_A) - eta_1**2 * kappa_A * np.sin(theta_A))
        x3 = 10.0 * (x_B - x_A) - (6.0 * eta_1 + 1.5 * eta_3) * np.cos(theta_A) \
             - (4.0 * eta_2 - 0.5 * eta_4) * np.cos(theta_B) \
             + 1.5 * eta_1**2 * kappa_A * np.sin(theta_A) \
             - 0.5 * eta_2**2 * kappa_B * np.sin(theta_B)
        x4 = -15.0 * (x_B - x_A) + (8.0 * eta_1 + 1.5*eta_3) * np.cos(theta_A) \
             + (7.0 * eta_2 - eta_4) * np.cos(theta_B) \
             - 1.5 * eta_1**2 * kappa_A * np.sin(theta_A) \
             + eta_2**2 * kappa_B * np.sin(theta_B)
        x5 = 6.0 * (x_B - x_A) - (3.0 * eta_1 + 0.5 * eta_3) * np.cos(theta_A) \
             - (3.0 * eta_2 - 0.5 * eta_4) * np.cos(theta_B) \
             + 0.5 * eta_1**2 * kappa_A * np.sin(theta_A) \
             - 0.5 * eta_2**2 * kappa_B * np.sin(theta_B)

        y0 = y_A
        y1 = eta_1 * np.sin(theta_A)
        y2 = 0.5 * (eta_3 * np.sin(theta_A) + eta_1**2 * kappa_A * np.cos(theta_A))
        y3 = 10.0 * (y_B - y_A) - (6.0 * eta_1 + 1.5 * eta_3) * np.sin(theta_A) \
             - (4.0 * eta_2 - 0.5 * eta_4) * np.sin(theta_B) \
             - 1.5 * eta_1**2 * kappa_A * np.cos(theta_A) \
             + 0.5 * eta_2**2 * kappa_B * np.cos(theta_B)
        y4 = -15.0 * (y_B - y_A) + (8.0 * eta_1 + 1.5 * eta_3) * np.sin(theta_A) \
             + (7.0 * eta_2 - eta_4) * np.sin(theta_B) \
             + 1.5 * eta_1**2 * kappa_A * np.cos(theta_A) \
             - eta_2**2 * kappa_B * np.cos(theta_B)
        y5 = 6.0 * (y_B - y_A) - (3.0 * eta_1 + 0.5 * eta_3) * np.sin(theta_A) \
             - (3.0 * eta_2 - 0.5 * eta_4) * np.sin(theta_B) \
             - 0.5 * eta_1**2 * kappa_A * np.cos(theta_A) \
             + 0.5 * eta_2**2 * kappa_B * np.cos(theta_B)

        X = np.array([x5, x4, x3, x2, x1, x0])
        XP = [5.0 * x5, 4.0 * x4, 3.0 * x3, 2.0 * x2, x1]
        XPP = [20.0 * x5, 12.0 * x4, 6.0 * x3, 2.0 * x2]

        Y = np.array([y5, y4, y3, y2, y1, y0])
        YP = [5.0 * y5, 4.0 * y4, 3.0 * y3, 2.0 * y2, y1]
        YPP = [20.0 * y5, 12.0 * y4, 6.0 * y3, 2.0 * y2]

        #u = 0.1
        u = walk_along_poly_path(X, Y, v*self.sim_dt)

        xp = np.polyval(XP, u)
        xpp = np.polyval(XPP, u)

        yp = np.polyval(YP, u)
        ypp = np.polyval(YPP, u)

        kappa = (xp * ypp - xpp * yp) / (xp**2 + yp**2)**(1.5)

        steering_angle = np.arctan(self.L * kappa)

        return steering_angle


def walk_along_poly_path(X_coeffs, Y_coeffs, desired_dist): # find inverse of arc length function
    du = 0.1

    arc_length = 0
    u = 0
    while arc_length < desired_dist:
        x0 = np.polyval(X_coeffs, u)
        y0 = np.polyval(Y_coeffs, u)

        x1 = np.polyval(X_coeffs, u + du)
        y1 = np.polyval(Y_coeffs, u + du)

        dist = hf.dist_points(x0, y0, x1, y1)
        arc_length += dist

        u += du

    # interpolate
    u -= du*(arc_length - desired_dist)/dist

    return u


















