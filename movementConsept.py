import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

############# create data set ###############
animation = True
count = [48, 24, 25, 7]  # Granny smith kleppe
# count = [25, 25, 25, 25]
sigma = 3  # mean and standard deviation
high = [1.4, 2.1, 2.8, 3.5, 4.2]
i = 0
data_list = []
row_l = 100
row_length = np.arange(0, row_l, 1.5)
for dim in row_length:
    for num in count:
        n = round(abs(np.random.normal(num, sigma)))
        xy_min = [dim, high[i]]
        xy_max = [dim + 1.5, high[i] + 0.7]
        data_1 = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
        data_list.append(data_1)
        i += 1
    i = 0
d = np.concatenate(data_list, axis=0)

plt.plot(d[:, 0], d[:, 1], '*r')
plt.show()


###############################################
############## simulation world################
class World:
    def __init__(self, data):
        self.data = data
        self.total_f = len(self.data)
        self.x_head = 0
        self.z1 = [[self.x_head + 4.5, high[0]], [self.x_head + 6, high[1]]]
        self.z3 = [[self.x_head + 3, high[1]], [self.x_head + 4.5, high[2]]]
        self.z5 = [[self.x_head + 1.5, high[2]], [self.x_head + 3, high[3]]]
        self.z7 = [[self.x_head, high[3]], [self.x_head + 1.5, high[4]]]
        self.waste = 0

    def update_zones(self):
        self.z1 = [[self.x_head + 4.5, high[0]], [self.x_head + 6, high[1]]]
        self.z3 = [[self.x_head + 3, high[1]], [self.x_head + 4.5, high[2]]]
        self.z5 = [[self.x_head + 1.5, high[2]], [self.x_head + 3, high[3]]]
        self.z7 = [[self.x_head, high[3]], [self.x_head + 1.5, high[4]]]

    def picked(self):
        return self.total_f - len(self.data)

    def get_fruits_in_box(self, min_a, max_a):
        inidx = np.all(np.logical_and(min_a <= self.data, self.data <= max_a), axis=1)
        inbox = self.data[inidx]
        return inbox

    def pick_in_zone(self, zone):
        p = self.get_fruits_in_box(zone[0], zone[1])
        if len(p) > 0:
            point = p[np.argmin(p[:, 0])]
            self.data = np.delete(self.data, np.where(self.data == point)[0], axis=0)
        else:
            self.waste = self.waste + 1


############# - regular method ################
class WilliMovingAlgorithm:
    def __init__(self, world_d):
        self.world = world_d
        self.z_c = self.world.z1
        self.willines = 2  # 2 ->50% of 4
        self.zones_targets = [100, 100, 100, 100]
        self.will_arr = [1, 1, 1, 1]

    def update_zones(self):
        self.world.update_zones()
        self.count_targets()

    def count_targets(self):
        self.zones_targets = [len(self.world.get_fruits_in_box(self.world.z1[0], self.world.z1[1])),
                              len(self.world.get_fruits_in_box(self.world.z3[0], self.world.z3[1])),
                              len(self.world.get_fruits_in_box(self.world.z5[0], self.world.z5[1])),
                              len(self.world.get_fruits_in_box(self.world.z7[0], self.world.z7[1]))]

    def move_controller(self):
        self.will_arr = np.array(self.zones_targets) > 0
        if np.sum(self.will_arr) <= self.willines:
            self.world.x_head = self.world.x_head + 1.5
            self.world.waste = self.world.waste + 4

    def pick_in_all_zones(self):
        self.world.pick_in_zone(self.world.z1)
        self.world.pick_in_zone(self.world.z3)
        self.world.pick_in_zone(self.world.z5)
        self.world.pick_in_zone(self.world.z7)


def plot_data(m_ct):
    fig, ax = plt.subplots(1)

    plt.plot(m_ct.world.data[:, 0], m_ct.world.data[:, 1], '*')
    z1p = Rectangle(tuple(m_ct.world.z1[0]), 1.5, 0.7, facecolor="red", edgecolor="black", alpha=0.3)
    z3p = Rectangle(tuple(m_ct.world.z3[0]), 1.5, 0.7, facecolor="red", edgecolor="black", alpha=0.3)
    z5p = Rectangle(tuple(m_ct.world.z5[0]), 1.5, 0.7, facecolor="red", edgecolor="black", alpha=0.3)
    z7p = Rectangle(tuple(m_ct.world.z7[0]), 1.5, 0.7, facecolor="red", edgecolor="black", alpha=0.3)

    ax.add_patch(z1p)
    ax.add_patch(z7p)
    ax.add_patch(z3p)
    ax.add_patch(z5p)

    print("watse time = " + str(m_ct.world.waste))
    print("total fruits = " + str(m_ct.world.total_f))
    print("picked = " + str(m_ct.world.picked()))
    print("percentage = " + str(m_ct.world.picked() / m_ct.world.total_f))


def update(frame):
    ax.clear()

    # Scatter plot for the data points
    ax.plot(frame['x'], frame['y'], '*', color='red', label='Data Points')

    # Add rectangles for zones
    for zone in frame['zones']:
        ax.add_patch(Rectangle(tuple(zone[0]), 1.5, 0.7, facecolor="blue", edgecolor="black", alpha=0.3))

    # Update title
    ax.set_title('Plot Data Animation')

    # Update layout
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    sys.stdout.write(
        "\rwaste time = {}\ttotal fruits = {}\tpicked = {}\tpercentage = {}".format(frame['waste'], frame['total_f'],
                                                                                    frame['picked'],
                                                                                    frame['picked'] / frame['total_f']))
    sys.stdout.flush()


# run algorithm2####
world2 = World(d)
m_ctrl2 = WilliMovingAlgorithm(world2)
frames = []
time = 0
while (m_ctrl2.world.x_head < row_l - 6) and (m_ctrl2.world.picked() < 4000):
    m_ctrl2.update_zones()
    m_ctrl2.pick_in_all_zones()
    m_ctrl2.move_controller()
    time = time + 1
    x = m_ctrl2.world.data[:, 0]  # Random x values
    y = m_ctrl2.world.data[:, 1]  # Random y values
    zones = [m_ctrl2.world.z1, m_ctrl2.world.z3, m_ctrl2.world.z5, m_ctrl2.world.z7]  # Example zones data
    waste = m_ctrl2.world.waste
    total_f = m_ctrl2.world.total_f
    picked = m_ctrl2.world.picked()
    frames.append({'x': x, 'y': y, 'zones': zones, 'waste': waste, 'total_f': total_f, 'picked': picked})
if animation:
    fig, ax = plt.subplots()

    # Create an animation
    ani = FuncAnimation(fig, update, frames=frames, interval=1000)
    plt.title('old_method')

    plt.show()

plot_data(m_ctrl2)
print("time = " + str(time))
print("rate =" + str(m_ctrl2.world.picked() / time))
# print("same time new method =" + str(new_rate * time))
