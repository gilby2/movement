import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# animation_speed in words: 100 -> 1 second per frame
SLOW = 1000
MEDIUM = 500
FAST = 100
VERY_FAST = 10
QUICK = 0.5

ZERO = 0
ONE = 1
TWO = 2
THREE = 3


SAMPLES_TO_SAVE = 1500

############# create data set ###############
animation = True
animation_speed = QUICK
fars_still_works = ZERO
count = [48, 24, 25, 7]  # Granny smith kleppe
# count = [25, 25, 25, 25]
sigma = 3  # mean and standard deviation
high = [1.4, 2.1, 2.8, 3.5, 4.2]
# high = [1.4, 1.8100230465036224, 2.222941426197958, 3.0098384382552834, 4.2]
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

# plt.plot(d[:, 0], d[:, 1], '*r')
# plt.show()


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
        self.collector = FruitHeightCollector()

    def update_zones(self):
        height_list = self.collector.zones_per_head_dict[self.x_head]
        self.z1 = [[self.x_head + 4.5, height_list[0]], [self.x_head + 6, height_list[1]]]
        self.z3 = [[self.x_head + 3, height_list[1]], [self.x_head + 4.5, height_list[2]]]
        self.z5 = [[self.x_head + 1.5, height_list[2]], [self.x_head + 3, height_list[3]]]
        self.z7 = [[self.x_head, height_list[3]], [self.x_head + 1.5, height_list[4]]]

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


class FruitHeightCollector:
    def __init__(self):
        self.heights = []
        self.zones_per_head_dict = {0: high}
        self.max_samples = SAMPLES_TO_SAVE

    def add_data(self, new_heights, zone):
        for height in new_heights:
            if len(self.heights) >= self.max_samples:
                self.heights.pop(0)  # Remove the oldest data point
            self.heights.append(height)

        print(f"Added {len(new_heights)} new samples from zone {zone}. "
              f"Total samples: {len(self.heights)}. "
              f"Space remaining: {self.max_samples - len(self.heights)}")

    def get_data(self):
        return self.heights

    def clear_data(self):
        self.heights = []
        print("All data cleared.")

    def divide_into_zones(self, head):
        if not self.heights:
            return []

        sorted_heights = sorted(self.heights)
        total_fruits = len(sorted_heights)
        fruits_per_zone = total_fruits // 4
        remainder = total_fruits % 4

        zones = [high[0]]
        start = 0
        for i in range(3):
            end = start + fruits_per_zone + (1 if i < remainder else 0)
            zones.append(sorted_heights[end])
            start = end
        zones.append(high[-1])

        self.zones_per_head_dict[head] = zones
        return


############# - regular method ################
class WilliMovingAlgorithm:
    def __init__(self, world_d):
        self.world = world_d
        self.z_c = self.world.z1
        self.willingness = fars_still_works  # 2 ->50% of 4
        self.zones_targets = [100, 100, 100, 100]
        self.will_arr = [1, 1, 1, 1]

    def update_zones(self):
        self.count_targets()

    def update_after_movement(self):
        self.world.collector.divide_into_zones(self.world.x_head)
        self.world.update_zones()
        self.count_targets_and_add_data()

    def count_targets_and_add_data(self):
        self.world.collector.add_data(self.world.get_fruits_in_box(self.world.z1[0], self.world.z1[1])[:, 1], 1)
        self.world.collector.add_data(self.world.get_fruits_in_box(self.world.z3[0], self.world.z3[1])[:, 1], 3)
        self.world.collector.add_data(self.world.get_fruits_in_box(self.world.z5[0], self.world.z5[1])[:, 1], 5)
        self.world.collector.add_data(self.world.get_fruits_in_box(self.world.z7[0], self.world.z7[1])[:, 1], 7)

    def count_targets(self):
        self.zones_targets = [len(self.world.get_fruits_in_box(self.world.z1[0], self.world.z1[1])),
                              len(self.world.get_fruits_in_box(self.world.z3[0], self.world.z3[1])),
                              len(self.world.get_fruits_in_box(self.world.z5[0], self.world.z5[1])),
                              len(self.world.get_fruits_in_box(self.world.z7[0], self.world.z7[1]))]

    def move_controller(self):
        self.will_arr = np.array(self.zones_targets) > 0
        if np.sum(self.will_arr) <= self.willingness:
            self.world.x_head = self.world.x_head + 1.5
            self.update_after_movement()
            self.world.waste = self.world.waste + 4

    def pick_in_all_zones(self):
        self.world.pick_in_zone(self.world.z1)
        self.world.pick_in_zone(self.world.z3)
        self.world.pick_in_zone(self.world.z5)
        self.world.pick_in_zone(self.world.z7)


def plot_data(m_ct):
    fig, ax = plt.subplots(1)

    plt.plot(m_ct.world.data[:, 0], m_ct.world.data[:, 1], '*')
    z1p = Rectangle(tuple(m_ct.world.z1[0]), 1.5, m_ct.world.z1[1][1]- m_ct.world.z1[0][1], facecolor="red", edgecolor="black", alpha=0.3)
    z3p = Rectangle(tuple(m_ct.world.z3[0]), 1.5, m_ct.world.z3[1][1]- m_ct.world.z3[0][1], facecolor="red", edgecolor="black", alpha=0.3)
    z5p = Rectangle(tuple(m_ct.world.z5[0]), 1.5, m_ct.world.z5[1][1]- m_ct.world.z5[0][1], facecolor="red", edgecolor="black", alpha=0.3)
    z7p = Rectangle(tuple(m_ct.world.z7[0]), 1.5, m_ct.world.z7[1][1]- m_ct.world.z7[0][1], facecolor="red", edgecolor="black", alpha=0.3)

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
        ax.add_patch(Rectangle(tuple(zone[0]), 1.5, zone[1][1] - zone[0][1], facecolor="blue", edgecolor="black", alpha=0.3))

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
m_ctrl2.count_targets_and_add_data()
while (m_ctrl2.world.x_head < row_l - 4.5) and (m_ctrl2.world.picked() < m_ctrl2.world.total_f):
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
    ani = FuncAnimation(fig, update, frames=frames, interval=animation_speed)
    plt.title('old_method')

    # Save the animation as an mp4 video
    # ani.save('picking.mp4', writer='ffmpeg')

    # Save the animation as a gif
    # ani.save('picking.gif', writer="pillow")

    plt.show()

# plot_data(m_ctrl2)
print("time = " + str(time))
print("rate =" + str(m_ctrl2.world.picked() / time))

pass
# print("same time new method =" + str(new_rate * time))
