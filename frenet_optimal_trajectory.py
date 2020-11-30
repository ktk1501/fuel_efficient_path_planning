"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import copy
import math
import sys
import os

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                "/../QuinticPolynomialsPlanner/")
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                "/../CubicSpline/")

try:
    from quintic_polynomials_planner import QuinticPolynomial
    import cubic_spline_planner
except ImportError:
    raise

SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 4.0  # maximum acceleration [m/ss] 2.0 FIXME
MAX_CURVATURE = 3.0  # maximum curvature [1/m] 1.0 FIXME
MAX_ROAD_WIDTH = 12.0  # maximum road width [m] 7.0 FIXME
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s] 1.2
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 25.0 / 3.6  # target speed [m/s]  30.0/3.6 FIXME
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 2  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]
GOAL_RADIUS = 1.0#5.0 

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = False


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
            self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(img, csp, c_speed, c_d, c_d_d, c_d_dd, s0, K_FUEL, K_SLOPE):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                #tfp = calc_global_paths(tfp, csp)
                

                # calc global positions
                for i in range(len(tfp.s)):
                    ix, iy = csp.calc_position(tfp.s[i])
                    if ix is None:
                        break
                    i_yaw = csp.calc_yaw(tfp.s[i])
                    di = tfp.d[i]
                    fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                    fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                    tfp.x.append(fx)
                    tfp.y.append(fy)
                
                tfp.s_height = [img[min(99,int(i))][min(99,int(j))] for (i,j) in zip(tfp.x, tfp.y)] # get intensities from image, which means height
                tfp.s_slope = np.asarray(tfp.s_height[1:], dtype="float16") - np.asarray(tfp.s_height[0:-1], dtype="float16") # subtract adjacent height to get slopes 
                # make tfp.s_slope 's unit as degree
                for idx in range(len(tfp.x)-1):
                    tfp.s_slope[idx] /= math.sqrt((tfp.x[idx]-tfp.x[idx+1])**2 + (tfp.y[idx]-tfp.y[idx+1])**2)

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                K_stationary = 1.7776  # Based on the ADD's report
                K_accel = 3.9914
                ACCEL_THRESH = 0.1
                s_dd_lenght = len(tfp.s_dd)
                # print(tfp.s_dd)
                Fs_stationary = np.count_nonzero(
                    np.array(tfp.s_dd) <= -ACCEL_THRESH)/s_dd_lenght
                Fs_accel = np.count_nonzero(
                    np.array(tfp.s_dd) > ACCEL_THRESH)/s_dd_lenght
                # Fuel consumption of Longitudinal motion planning
                Fs = (K_stationary * Fs_stationary + K_accel *
                      Fs_accel) / (K_stationary + K_accel)
                '''
                print("Fs_stationary: %.2f,  Fs_accel: %.2f " %
                      (Fs_stationary, Fs_accel))
                '''
                #print(s_slope)
                Fslope = [ max(0, 3 + slope * 0.8) for slope in tfp.s_slope]
                Fslope = sum(Fslope)
                #print(Fslope)


                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2
                K_F = 30.0
                K_S = 0.1

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cfuel = K_F * Fs
                tfp.cslope = K_S * Fslope
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv + K_FUEL * tfp.cfuel + K_SLOPE * tfp.cslope

                '''
                print("K_J*Jp: %.2f,  K_T*Ti: %.2f,  K_D * tfp.d[-1] ** 2: %.2f" % (
                    K_J * Jp, K_T * Ti, K_D * tfp.d[-1] ** 2))
                print("K_J*Js: %.2f,  K_T*Ti: %.2f,  K_D * ds: %.2f, K_Fuel * Fs: %.2f" %
                      (K_J * Js, K_T * Ti, K_D * ds, Fs*K_Fuel))
                '''
                #print("LAT Cost: %.2f,  LON Cost: %.2f, Fuel Cost: %.2f, Slope Cost: %.2f" % (K_LAT * tfp.cd, K_LON * tfp.cv, K_FUEL * tfp.cfuel, tfp.cslope))

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            if fp.ds[i] == 0:
                fp.ds[i] = 0.0001
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(img, csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, K_FUEL, K_SLOPE):
    fplist = calc_frenet_paths(img, csp, c_speed, c_d, c_d_d, c_d_dd, s0, K_FUEL, K_SLOPE)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def simulation( img, wx, wy, ob, k_fuel, k_slope, prev_paths_x = None, prev_paths_y = None):
    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    total_fuel = 0.0
    paths_x = []  # store vehicle trajectory
    paths_y = []

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    area = 20.0  # animation area length [m]

    # for animation
    temp_ob = np.array(ob)

    

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(img,
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, K_FUEL=k_fuel, K_SLOPE=k_slope)

        if path == None:
            continue
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]
        c_stationary = abs(min(c_accel, 0))
        c_fuel = path.cfuel + path.cslope
        total_fuel += c_fuel
        print("%.2f" % total_fuel)
        

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= GOAL_RADIUS:
            print("Goal")
            break

        if show_animation:  # pragma: no cover      
            plt.close()
            plt.clf()
            # Surface plot
            xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            fig = plt.figure(figsize=(15, 15))
            ax = fig.gca(projection='3d')
            ax.plot_surface(xx, yy, img, rstride=4, 
                    cstride=4,cmap=plt.cm.gray, linewidth=2, alpha=0.6)
            ax.set_zlim3d(0,256)

            # object 3D plot
            height = [img[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(paths_x,paths_y)]
            ax.plot(paths_x,paths_y,height, alpha=1, marker='v', c='c')

            paths_x.append(path.x[1])
            paths_y.append(path.y[1])
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            #plt.subplot(2, 1, 1)

            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(paths_x, paths_y, "vb")


            # 3D plot
            height = [img[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(paths_x,paths_y)]
            ax.plot(paths_x,paths_y,height, alpha=1, marker='v', c='b')
            # obstacles
            height = [img[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(ob[:,0],ob[:,1])]
            ax.scatter(ob[:,0],ob[:,1],height, alpha=1, marker='x', c='k')
            # desired path
            height = [img[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(tx,ty)]
            ax.plot(tx,ty,height, alpha=1)

            if prev_paths_x is not None:
                plt.plot(prev_paths_x, prev_paths_y, "*c")
                # 3D plot
                height = [img[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(prev_paths_x,prev_paths_y)]
                ax.plot(prev_paths_x,prev_paths_y,height, alpha=0.7, marker='*', c='c')
            plt.xlim(path.x[1] - 2*area, path.x[1] + 2*area)
            plt.ylim(path.y[1] - 2*area, path.y[1] + 2*area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4] +
                      "    a[km/h2]:" + str(c_accel * 3.6)[0:4] +
                      "    stationary[km/h2]:" + str(c_stationary * 3.6)[0:4])
            plt.pause(0.0001)
    return paths_x, paths_y

def main():
    print(__file__ + " start!!")
    img = cv2.imread('terrain2.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 50.5, 55, 70, 90]
    wy = [0.0, 6.0, 15.0, 26.5, 10.0, 52.5, 85.5, 90]
    # obstacle lists
    ob = np.array([[20.0, 10.0],
                   [20.0, 16.0],
                   [23.0, 18.0],
                   [35.0, 28.0],
                   [34.0, 30.0],
                   [38.0, 22.0],
                   [40.0, 31.0],
                   [40.0, 25.0],
                   [50.0, 13.0],
                   [53.0, 48.0],
                   [54.0, 54.0],
                   [54.0, 49.0],
                   [55.0, 60.0],
                   [58.0, 65.0],
                   [67.0, 85.0],
                   [84.0, 86.0],
                   ])

    paths_1_x, paths_1_y = simulation(img=img_gray, wx=wx, wy=wy, ob=ob, k_fuel = 0.0, k_slope = 0.0)

    paths_2_x, paths_2_y = simulation(img=img_gray, wx=wx, wy=wy, ob=ob, k_fuel = 1.0, k_slope = 1.0, prev_paths_x = paths_1_x, prev_paths_y = paths_1_y)

    print("Finish")
    if show_animation:  # pragma: no cover
        # plt.grid(True)
        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
        plt.close()
        plt.clf()
        # Surface plot
        xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        fig = plt.figure(figsize=(15, 15))
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, img_gray, rstride=4, 
                cstride=4,cmap=plt.cm.gray, linewidth=2, alpha=0.6)
        ax.set_zlim3d(0,256)

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        #plt.subplot(2, 1, 1)

        plt.plot(tx, ty)
        plt.plot(ob[:, 0], ob[:, 1], "xk")
        #plt.plot(path.x[1:], path.y[1:], "-or")
        plt.plot(paths_1_x, paths_1_y, "vb")


        # 3D plot
        height = [img_gray[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(paths_1_x,paths_1_y)]
        ax.plot(paths_1_x,paths_1_y,height, alpha=0.8, marker='v', c='b')
        # obstacles
        height = [img_gray[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(ob[:,0],ob[:,1])]
        ax.scatter(ob[:,0],ob[:,1],height, alpha=1, marker='x', c='k')
        # desired path
        height = [img_gray[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(tx,ty)]
        ax.plot(tx,ty,height, alpha=0.6)

        plt.plot(paths_2_x, paths_2_y, "*c")
        # 3D plot
        height = [img_gray[min(99,int(i))][min(99,int(j))]+2 for (i,j) in zip(paths_2_x,paths_2_y)]
        ax.plot(paths_2_x,paths_2_y,height, alpha=0.8, marker='*', c='c')

        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
