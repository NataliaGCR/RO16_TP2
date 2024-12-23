"""
RRT_2D
@author: huiming zhou

Modified by David Filliat
"""

import os
import sys
import math
import numpy as np
import plotting, utils
import env
import time

# parameters
showAnimation = False

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Rrt:
    def __init__(self, environment, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = environment
        if showAnimation:
            self.plotting = plotting.Plotting(self.env, s_start, s_goal)
        self.utils = utils.Utils(self.env)

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        iter_goal = None
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and iter_goal == None and not self.utils.is_collision(node_new, self.s_goal):
                    node_new = self.new_state(node_new, self.s_goal)
                    node_goal = node_new
                    iter_goal = i

        if iter_goal == None:
            return None, self.iter_max
        else:
            return self.extract_path(node_goal), iter_goal

    def generate_random_node(self, goal_sample_rate):
        if np.random.random() < goal_sample_rate:
            return self.s_goal

        delta = self.utils.delta

        node = Node((
                np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)
            ))
        p = np.random.random()

        delta *= 8
        
        if p < 0.8: 
            while True:
                id = np.random.randint(len(self.env.obs_rectangle)) 
                x, y, w, h = self.env.obs_rectangle[id] 
                node_list = [
                    Node((np.random.uniform(x - delta, x), np.random.uniform(y - delta, y + h + delta))),
                    Node((np.random.uniform(x + w, x + w + delta), np.random.uniform(y - delta, y + h + delta))),
                    Node((np.random.uniform(x - delta, x + w + delta), np.random.uniform(y - delta, y))),
                    Node((np.random.uniform(x - delta, x + w + delta), np.random.uniform(y + h, y + h + delta)))
                ]
                
                node = node_list[np.random.randint(len(node_list))]
                if not self.utils.is_inside_obs(node):
                    break
        return node




    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def get_path_length(path):
    """
    Compute path length
    """
    length = 0
    for i,k in zip(path[0::], path[1::]):
        length += np.linalg.norm(np.array(i) - np.array(k)) # math.dist(i,k)
    return length

'''
def main():
    x_start=(2, 2)  # Starting node
    x_goal=(49, 24)  # Goal node
    environment = env.Env2()
    itermax = 1500
    step_len = 2

    start_time = time.time()
    rrt = Rrt(environment, x_start, x_goal, step_len, 0.10, itermax)
    path, nb_iter = rrt.planning()
    end_time = time.time()

    execution_time = end_time - start_time

    if path:
        print('Found path in ' + str(nb_iter) + ' iterations, length : ' + str(get_path_length(path)))
        print('Execution time: {:.2f} seconds'.format(execution_time))

        #if showAnimation:
            #rrt.plotting.animation(rrt.vertex, path, "RRT", True)
            #plotting.plt.show()

    else:
        print("No Path Found in " + str(nb_iter) + " iterations!")
        print('Execution time: {:.2f} seconds'.format(execution_time))

        #if showAnimation:
            #rrt.plotting.animation(rrt.vertex, [], "RRT", True)
            #plotting.plt.show()

'''

#QUESTION 4        
def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node
    environment = env.Env2()
    itermax = 1500
    step_len = 2

    corner_sample_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Vary corner sampling rate
    results = []

    for corner_rate in corner_sample_rates:
        execution_times = []
        iterations = []
        path_lengths = []
        failures = 0
        num_trials = 50

        for _ in range(num_trials):
            start_time = time.time()
            rrt = Rrt(environment, x_start, x_goal, step_len, 0.10, itermax)
            rrt.goal_sample_rate = corner_rate
            path, nb_iter = rrt.planning()
            end_time = time.time()

            execution_time = end_time - start_time
            execution_times.append(execution_time)

            if path:
                path_lengths.append(get_path_length(path))
                iterations.append(nb_iter)
            else:
                failures += 1

        # Calculate average and standard deviation results
        avg_execution_time = np.mean(execution_times)
        std_execution_time = np.std(execution_times)

        avg_iterations = np.mean(iterations) if iterations else 0
        std_iterations = np.std(iterations) if iterations else 0

        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        std_path_length = np.std(path_lengths) if path_lengths else 0

        failure_rate = failures / num_trials * 100

        results.append({
            "corner_sample_rate": corner_rate,
            "failure_rate": failure_rate,
            "avg_iterations": avg_iterations,
            "std_iterations": std_iterations,
            "avg_path_length": avg_path_length,
            "std_path_length": std_path_length,
            "avg_execution_time": avg_execution_time,
            "std_execution_time": std_execution_time
        })

    # Print results
    print("Results:")
    for result in results:
        print(f"Corner Sample Rate: {result['corner_sample_rate'] * 100:.0f}%")
        print(f"  Failure Rate: {result['failure_rate']:.2f}%")
        print(f"  Average Iterations: {result['avg_iterations']:.2f} ± {result['std_iterations']:.2f}")
        print(f"  Average Path Length: {result['avg_path_length']:.2f} ± {result['std_path_length']:.2f}")
        print(f"  Average Execution Time: {result['avg_execution_time']:.2f} ± {result['std_execution_time']:.2f} seconds")

if __name__ == "__main__":
    main()
