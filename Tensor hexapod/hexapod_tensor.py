import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


class Hexapod:
    def __init__(self, axis):
        self.axis = axis
        self.alpha = 30.
        self.beta = 15.
        self.L = 1.5
        self.h_c = 2.5
        self.r = 1.
        self.m = 2500.

        self.A_0 = np.round([[self.r*np.sin(2*np.pi/3*i + np.pi),
                              self.r*np.cos(2*np.pi/3*i + np.pi),
                              -self.h_c] for i in range(-1, 2)], 5)
        self.B = np.array([])
        self.A = np.array([])
        self.all_full_lengths = np.array([])

        self.nu = 1.
        self.fi_x_0 = 3.
        self.fi_x = lambda t: self.fi_x_0 * np.sin(2*np.pi * self.nu * t)
        self.prime_fi_x = lambda t: self.fi_x_0 * 2*np.pi * self.nu * np.cos(2*np.pi * self.nu * t)
        self.fi_y_0 = 5.
        self.fi_y = lambda t: self.fi_y_0 * np.sin(2*np.pi * self.nu * t)
        self.prime_fi_y = lambda t: self.fi_y_0 * 2*np.pi * self.nu * np.cos(2*np.pi * self.nu * t)

        self.J = np.array([[5000, 0, 0],
                           [0, 5000, 0],
                           [0, 0, 3500]], np.float32)

        self.H = np.cos(np.pi/180. * self.beta) * np.cos(np.pi/180. * self.alpha/2) * self.L
        self.R_b = (self.L**2 - self.H**2) ** 0.5

        self.R_matrix_y = lambda t: np.round([[1, 0, 0],
                                              [0, np.cos(self.fi_y(t)*np.pi/180.), -np.sin(self.fi_y(t)*np.pi/180.)],
                                              [0, np.sin(self.fi_y(t)*np.pi/180.), np.cos(self.fi_y(t)*np.pi/180.)]], 5)

        self.R_matrix_x = lambda t: np.round([[np.cos(self.fi_x(t)*np.pi/180.), -np.sin(self.fi_x(t)*np.pi/180.), 0],
                                              [np.sin(self.fi_x(t)*np.pi/180.), np.cos(self.fi_x(t)*np.pi/180.), 0],
                                              [0, 0, 1]], 5)

        self.R_matrix_z = lambda t: np.round([[np.cos(self.fi_y(t)*np.pi/180.), 0, np.sin(self.fi_y(t)*np.pi/180.)],
                                              [0, 1, 0],
                                              [-np.sin(self.fi_y(t)*np.pi/180.), 0, np.cos(self.fi_y(t)*np.pi/180.)]], 5)
        # для построения точек B
        self.h = self.L * np.cos(np.pi/180.*self.alpha/2) * np.sin(np.pi/180.*self.beta)
        self.a = self.L * np.sin(np.pi/180.*self.alpha/2)
        self.r = (self.h**2 + self.a**2)**0.5

        self.end_time = 2.
        self.start_time = 0.
        self.steps = 100

    def set_B(self):
        for i, A in enumerate(self.A_0):
            a = A[:2]
            b1 = np.array([self.h, self.a])
            b2 = np.array([self.h, -self.a])

            kappa = np.array([[np.cos(np.pi / 180 * (30-120*i)), -np.sin(np.pi / 180 * (30-120*i))],
                              [np.sin(np.pi / 180 * (30-120*i)), np.cos(np.pi / 180 * (30-120*i))]])

            p1 = np.dot(kappa, b1) + a
            p2 = np.dot(kappa, b2) + a
            p1 = np.append(p1, -self.H - self.h_c)
            p2 = np.append(p2, -self.H - self.h_c)
            self.B = np.hstack((self.B, p1))
            self.B = np.hstack((self.B, p2))

        self.B = self.B.reshape(6, 3)

        # test
        i = 0
        for A in self.A_0:
            assert np.linalg.norm(np.subtract(A, self.B[i])) - self.L <= 1e-4
            assert np.linalg.norm(np.subtract(A, self.B[i + 1])) - self.L <= 1e-4
            print(np.linalg.norm(np.subtract(A, self.B[i])))
            print(np.linalg.norm(np.subtract(A, self.B[i + 1])))
            i += 2

    def plot_top_plane(self):
        # print(self.B)
        A = np.dot(self.A_0, self.R_matrix_z(.25))
        df_A = pd.DataFrame(data=self.A_0, columns=['x', 'y', 'z'])
        df_B = pd.DataFrame(data=self.B, columns=['x', 'y', 'z'])
        df_test = pd.DataFrame(data=A, columns=['x', 'y', 'z'])
        # print(self.A_0)

        # plt.figure(figsize=(12, 10))
        plt.grid()
        plt.scatter(df_A.x.values, df_A.y.values)
        plt.scatter(df_B.x.values, df_B.y.values)
        plt.plot(df_test.x.values, df_test.y.values)

        # ax = plt.axes(projection='3d')

        # ax.plot(df_A.x.values, df_A.y.values, df_A.z.values, '-b')
        # ax.scatter(df_B.x.values, df_B.y.values, df_B.z.values, '-g')
        # ax.plot(df_test.x.values, df_test.y.values, df_test.z.values, '-r')
        plt.show()

    def get_delta_l(self):
        R_matrix = None
        if self.axis == 'x':
            R_matrix = self.R_matrix_x
        elif self.axis == 'y':
            R_matrix = self.R_matrix_y
        elif self.axis == 'z':
            R_matrix = self.R_matrix_z

        indexes = [[0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 5]]

        L_all = []  # удлинения каждого цилиндра за период
        coordinates_A = []  # координаты
        for i, j in indexes:
            print('index', i, j)
            dl = []
            coord = []
            for t in np.linspace(self.start_time, self.end_time, self.steps):
                try:
                    A = np.dot(R_matrix(t), self.A_0[i])
                except Exception:
                    print('Type error axis')

                L = np.sum((A - self.B[j])**2)**0.5
                L_0 = np.sum((self.A_0[i] - self.B[j])**2)**0.5
                assert L_0 - self.L <= 1e-4

                print('dL[мм] = {:.5f}'.format((L - L_0) * 1e4))
                dl.append(round(((L - L_0) * 1e4), 5))
                coord.append(A)

            coordinates_A.append(coord)
            L_all.append(dl)

            # численно находим СКОРОСТЬ изменения длины цилиндра
            v = []
            time = np.linspace(self.start_time, self.end_time, self.steps)
            for k in range(99):
                v.append((dl[k + 1] - dl[k]) / (time[k + 1] - time[k]))

            print('v = ', np.round(v, 3))
            print('v =', (np.max(np.abs(dl))) / self.end_time)
            print('###########################################################################')

        coordinates_A = coordinates_A[0::2]  # исключим повторение вершни
        self.A = coordinates_A

        fig = plt.figure(figsize=(12,10))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        colors = {0: 'r+--', 1: 'rx-',
                  2: 'g+--', 3: 'gx-',
                  4: 'b+--', 5: 'bx-'}
        for i in range(6):
            print(L_all[i])
            plt.plot(np.linspace(self.start_time, self.end_time, self.steps), L_all[i], colors[i])

        axes.set_xlabel('time')
        axes.set_ylabel('$/delta L$')
        axes.legend([r'1 line', '2 line', '3 line', '4 line', '5 line', '6 line'], loc=0)
        plt.grid()
        plt.show()

        self.plot_3d_lines(coordinates_A)
        # self.plot_animate(coordinates_A)

    def plot_3d_lines(self, A):
        indexes = [[0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 5]]

        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')

        # Make a 3D quiver plot
        x, y, z = np.zeros((3, 3))
        u, v, w = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

        # ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1)
        colors = {0: 'r', 1: 'orange',
                  2: 'g', 3: 'olive',
                  4: 'b', 5: 'navy'}
        markers = {0: '^', 1: '^',
                   2: 'o', 3: 'o',
                   4: '*', 5: '*'}

        # задать легенду
        for i, j in indexes:
            df_A = pd.Series(data=self.A_0[i], index=['x', 'y', 'z'])
            df_B = pd.Series(data=self.B[j], index=['x', 'y', 'z'])
            x = [df_A.x, df_B.x]
            y = [df_A.y, df_B.y]
            z = [df_A.z, df_B.z]
            ax.scatter(x, y, z, c=colors[j], marker=markers[j], s=20.)
        ax.legend([r'1', '2', '3', '4', '5', '6'], loc=0)

        # indexes = [[0, 0], [1, 2], [2, 4]]
        # построить смещения каждого поршня
        for i, j in indexes:
            for k, a in enumerate(A[i]):
                df_A = pd.Series(data=a, index=['x', 'y', 'z'])
                df_B = pd.Series(data=self.B[j], index=['x', 'y', 'z'])
                x = [df_A.x, df_B.x]
                y = [df_A.y, df_B.y]
                z = [df_A.z, df_B.z]
                if k % 30 == 0:
                    ax.plot(x, y, z, c=colors[j], marker=markers[j])
                    print('H_A =', z[0])

        cur_A = np.array(A)
        # посторить смещение верхней плтаформы
        for i in range(0, self.steps, 8):
            a = np.array([cur_A[0, i], cur_A[1, i], cur_A[2, i]])
            df_A = pd.DataFrame(data=a, columns=['x', 'y', 'z'])
            df_A = pd.concat((df_A, df_A.take([0])), axis=0)
            ax.plot(df_A.x.values, df_A.y.values, df_A.z.values, c='gray')

        # отрисовать начальные положения верхней и нижней платформы
        df_B = pd.DataFrame(data=self.B, columns=['x', 'y', 'z'])
        df_B = pd.concat((df_B, df_B.take([0])))
        df_A = pd.DataFrame(data=self.A_0, columns=['x', 'y', 'z'])
        df_A = pd.concat((df_A, df_A.take([0])), axis=0)

        ax.plot(df_B.x.values, df_B.y.values, df_B.z.values, c='black', linewidth=4.)
        ax.plot(df_A.x.values, df_A.y.values, df_A.z.values, c='black', linewidth=4.)

        ax.view_init(30, -39)
        plt.show(block=False)

    def get_all_full_len(self):
        """
        Полная длина каждого цилиндра
        :return: вернуть список списков - для каждого момента времени записаны длины каждого цилиндра
        """
        indexes = [[0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 5]]
        self.get_delta_l()
        all_full_len = []
        for t in np.linspace(self.start_time, self.end_time, self.steps):
            line = []
            for i, j in indexes:
                L = np.sum((np.dot(self.R_matrix_y(t), self.A_0[i]) - self.B[j]) ** 2) ** 0.5
                line.append(L)
            all_full_len.append(line)

        self.all_full_lengths = np.array(all_full_len)

    def solve_dynamic_forces(self):
        """
        решение обратной задачи стенда
        Первое приближение  - решение двумерной зазадчи для пооврота оси вокруг оси х
        :return: минимальная и максимальная нагрузка на каждый цилиндр
        """
        self.get_all_full_len()
        all_forces = []  # силы всех цилиндров в каждый момент времени
        time = np.linspace(self.start_time, self.end_time, self.steps)
        for lenghts, t in zip(self.all_full_lengths, time):
            l1, l2, l3, l4, l5, l6 = lenghts
            print("lengths: ", (lenghts - self.L)*1e4)
            try:
                f1 = (self.J[1, 1]*self.prime_fi_y(t)/np.cos(np.pi/180.*self.beta) -
                      self.m*10*l3/np.sin(np.pi/180.*self.beta)) / (l1 - l2) / 3

                f2 = (self.J[1, 1]*self.prime_fi_y(t)/np.cos(np.pi/180.*self.beta) -
                      self.m*10*l1/np.sin(np.pi/180.*self.beta)) / (l2 - l1) / 3

                all_forces.append([f1, f2])
            except Exception:
                continue

        all_forces = np.array(all_forces)
        print(all_forces)


    def plot_animate(self, A):
        """"
        try to create animate function to plot mechanisms
        """
        fig = plt.figure()
        fig.set_tight_layout(False)
        ax = plt.axes(projection='3d')
        global cnt
        cnt = ax

        global cur_A
        global cur_B
        cur_A = A[0]
        cur_B = self.B[0]

        def steps(count=1):
            for i in range(count):
                df_A = pd.Series(data=cur_A[i], index=['x', 'y', 'z'])
                df_B = pd.Series(data=cur_B, index=['x', 'y', 'z'])
                x = [df_A.x, df_B.x]
                y = [df_A.y, df_B.y]
                z = [df_A.z, df_B.z]
                cnt.plot(x, y, z)

        def animate(frame):
            steps(1)
            return cnt

        anim = animation.FuncAnimation(fig, animate, frames=100)
        plt.show()


if __name__ == "__main__":
    hex = Hexapod(axis='z')
    hex.set_B()
    # hex.plot_top_plane()
    hex.get_delta_l()
    hex.solve_dynamic_forces()
    """
    Заметки на полях:
    
    Вращние вокруг оси х:
        симметрия длин по
            1) 1, 3, 4, 6
            2) 2, 5
            3) 3, 4
    """