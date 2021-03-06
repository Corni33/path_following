import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.interpolate import splprep, splev

from simulator import Simulator
from controller import *

# define driving path points
# xp = [0, 20, 40, 55, 80, 100] # oscillating path
# yp = [0, 10, 0, 20, 0, 10]

xp = [0, 20, 40, 55, 70, 70] # path with bend at the end
yp = [0, 10, 8, 20, 10, -10]

# xp = [0, 20, 40, 200] # straight path
# yp = [0, 0, 0, 0]

# additional parameters
L = 3.0 # wheelbase (dist between front and rear axle)
sim_dt = 0.1 # simulation step size
num_sim_steps = 80

controller_type = FLATNESS_CONTROLLER
#controller_type = PURE_PURSUIT_CONTROLLER

# controller look ahead distance
look_ahead_dist = 10

# additional parameters of flatness controller
eta_1 = 10.0 # eta_1 and eta_2 are "velocity parameters"
eta_2 = 10.0
eta_3 = 10.0 # eta_3 and eta_4 are "shape parameters"
eta_4 = 50.0

# ------------------------------------------------------------
# ------------------------------------------------------------

# create smooth B-spline representation of the driving path
path_spline, knots = splprep([xp, yp], s=2.0)
x_path, y_path = splev(np.linspace(0, 1, 100), path_spline)

# prepare plot window
fig = plt.figure(figsize=(8,8))

plt.subplot(211)
plt.subplots_adjust(left=0.1, bottom=0.35, top=0.95)
plot_line_steering, = plt.plot([], [])
plt.axis([0, 20, -1.2, 1.2])
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('steering angle [rad]')
axis_steering_angle = plt.gca()
axis_steering_angle.set_title('steering angle')

plt.subplot(212)
plt.plot(x_path, y_path, '--', lw=1)
plt.plot(xp, yp, 'bo', ms=2)
plot_line_trajectory, = plt.plot([], [], lw=1, color='red')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.grid()

slider_bg_color = (0.95, 0.95, 0.95)
ax_look_dist = plt.axes([0.2, 0.25, 0.7, 0.03], facecolor=slider_bg_color)
ax_eta_1 = plt.axes([0.2, 0.2, 0.7, 0.03], facecolor=slider_bg_color)
ax_eta_2 = plt.axes([0.2, 0.15, 0.7, 0.03], facecolor=slider_bg_color)
ax_eta_3 = plt.axes([0.2, 0.1, 0.7, 0.03], facecolor=slider_bg_color)
ax_eta_4 = plt.axes([0.2, 0.05, 0.7, 0.03], facecolor=slider_bg_color)

s_look_dist = Slider(ax_look_dist, 'look_ahead_dist', 0.0, 100.0, valinit=look_ahead_dist)
s_eta_1 = Slider(ax_eta_1, 'eta_1', 0.0, 100.0, valinit=eta_1)
s_eta_2 = Slider(ax_eta_2, 'eta_2', 0.0, 100.0, valinit=eta_2)
s_eta_3 = Slider(ax_eta_3, 'eta_3', -200, 200.0, valinit=eta_3)
s_eta_4 = Slider(ax_eta_4, 'eta_4', -200.0, 200.0, valinit=eta_4)


if controller_type == FLATNESS_CONTROLLER:
    con = FlatnessController()
    fig.canvas.set_window_title('Flatness Controller')
elif controller_type == PURE_PURSUIT_CONTROLLER:
    con = PurePursuitController()
    fig.canvas.set_window_title('Pure Pursuit Controller')
else:
    con = None

con.init(sim_dt=sim_dt, L=L)

sim = Simulator()
sim.init(path_spline, num_points=100, dt=sim_dt, L=L)


def update(val):

    look_ahead_dist_ = s_look_dist.val
    eta_ = [s_eta_1.val, s_eta_2.val, s_eta_3.val, s_eta_4.val]

    steering_angle = 0
    steering_angle_list = []

    sim.reset([-1, 3, 0, 10])

    for t_step in range(num_sim_steps):

        look_ahead_point = sim.look_ahead(look_ahead_dist_) # returns: (x_look, y_look, theta, kappa)

        steering_angle = con.get_steering_angle(sim.state,
                                                steering_angle,
                                                look_ahead_point,
                                                eta_)

        sim.move(steering_angle)

        steering_angle_list.append(steering_angle)

    sim.finalize()

    plot_line_trajectory.set_xdata(sim.sim_solution[:, 0])
    plot_line_trajectory.set_ydata(sim.sim_solution[:, 1])

    plot_line_steering.set_xdata(sim.sim_time_vector[:-1])
    plot_line_steering.set_ydata(steering_angle_list)
    axis_steering_angle.axis([0, sim.sim_time_vector[-1], -1.2, 1.2])

    fig.canvas.draw_idle()


s_eta_1.on_changed(update)
s_eta_2.on_changed(update)
s_eta_3.on_changed(update)
s_eta_4.on_changed(update)
s_look_dist.on_changed(update)

plt.show()
