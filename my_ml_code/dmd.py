# math packages
import numpy as np
from scipy import linalg as la
from cmath import exp

# data manipulation
import pandas as pd

# visualization packages
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation


# operating system
import os


class manipulate:

    '''
    This class contains helpful functions for the specific matrix
    manipulation needed for Dynamic Mode Decomposition.
    '''

    def two_2_three(A, n, verbose=False):
        '''
        This function will take a matrix that has three dimensional data
        expressed as a 2D matrix (columns as full vectors) and return a 3D
        matrix output where the columns are transformed back into 2D matrices
        and then placed in a numpy list of matricies.

        This is useful for DMD
        computation because the general input to DMD is a 2D matrix, but the
        output may need to be converted back to 3D in order to visualize
        results.

        This function is also useful for the visualization of dynamic
        modes that are computed by DMD.

        inputs:
        A - 2D matrix where columns represent (mxn) matricies
        n - # of columns in each smaller matrix

        outputs:
        A_3 - 3D numpy array containing each indiviual mxn matrix

        Options:
        verbose - prints matricies in order to see the transformation
                  that takes place
        '''
        if verbose:
            print('Entering 2D to 3D conversion:\n')

        # make sure the input is a numpy array
        A = np.array(A)
        rows, cols = np.shape(A)

        # calculate the number of columns in the matrix and quit if
        # the numbers do not match
        m = int(rows / n)
        if rows % m != 0:
            print('Invalid column input. \nFunction Exited.')
            return None

        # initalize A_3, the matrix
        A_3 = np.zeros((cols, m, n))

        # now loop over the columns construct the matrices and place them
        # in the A_3 matrix
        a_hold = np.zeros((m, n))
        for i in range(cols):
            # grab the column that holds the vector
            col_hold = A[:, i]

            # convert the column into the holder matrix shape
            for j in range(n):
                a_hold[:, j] = col_hold[j * m:(j + 1) * m]

            # place the matrix in the new 3D holder
            A_3[i] = a_hold

        # print the transformation that occured
        if verbose:
            print('A =\n', A, '\n')
            print('A_3 =\n', A_3, '\n')

        # the program is finished
        return A_3

    def three_2_two(A, verbose=False):
        '''
        This function will take a matrix that has three dimensional data
        expressed as a 3D matrix and return a 2D matrix output where the
        columns represent the matricies in the original 3D matrix

        This is neccessary for DMD computation because spacial-temporal
        data has to be converted into one single matrix in order to perform
        the neccessary singluar value decomposition.

        inputs:
        A - 3D matrix where columns represent (mxn) matricies

        outputs:
        A_2 - 2D numpy array containing each indiviual mxn matrix as a
              column

        Options:
        verbose - prints matricies in order to see the transformation
                  that takes place
        '''

        if verbose:
            print('Entering 3D to 2D conversion:\n')

        # make sure the input is a numpy array and store shape
        A = np.array(A)
        length, rows, cols = np.shape(A)
        n = length  # number of columns for A_2

        # calculate the length of each column needed (number of rows)
        m = rows * cols

        # initalize A_2, the matrix to be returned
        A_2 = np.zeros((m, n))

        # now loop over the matricies, construct the columns and place them
        # in the A_2 matrix
        vec = np.zeros(m)
        A_2 = A_2.transpose()
        for i in range(n):
            # grab the matrix
            matrix = A[i]

            # loop through the columns and store them in a_hold
            for j in range(cols):
                vec[j * rows:(j + 1) * rows] = matrix[:, j]

            # place the matrix in the new 3D holder
            A_2[i] = vec
        A_2 = A_2.transpose()
        # print the transformation that occured
        if verbose:
            print('A =\n', A, '\n')
            print('A_2 =\n', A_2, '\n')

        # the program is finished
        return A_2

    def split(Xf, verbose=False):
        '''
        This function will perform a crutical manipulation for DMD
        which is the splitting of a spacial-temporal matrix (Xf) into
        two matrices (X and Xp). The X matrix is the time series for
        1 to n-1 and Xp is the time series of 2 to n where n is the
        number of time intervals (columns of the original Xf).

        input:
        Xf - matix of full spacial-temporal data

        output:
        X - matix for times 1 to n-1
        Xp - matix for times 2 to n

        options:
        verbose - boolean for visualization of splitting
        '''

        if verbose:
            print('Entering the matrix splitting function:')

        if verbose:
            print('Xf =\n', Xf, '\n')

        X = Xf[:, :-1]
        Xp = Xf[:, 1:]

        if verbose:
            print('X =\n', X, '\n')
            print('Xp =\n', Xp, '\n')
        return X, Xp


class examples:

    '''
    This class will hold functions that will give very simple examples of
    how DMD works in this library of functions. There will be theoretical
    examples as well as data driven examples in this class once the class
    is out of development
    '''

    def kutz():
        '''
        This is a simple example of how DMD can be used to reconstruct
        a complex example from Kutz's book on DMD.
        '''

        print('To show how DMD can be performed using the class given')
        print('let us take a look at an example from Kutz\'s book on DMD\n')

        print('We will look at a complex, periodic function given below:\n')

        print('f(x,t) = sech(x+3)exp(2.3it) + 2sech(x)tanh(x)exp(2.8it)\n')

        print('Now, the 3D function will be plotted on a surface plot as well as its')
        print('DMD reconstruction based on rank reduction at 1,2, and 3 singular values.\n')

        print('It can be shown that this function only has rank = 2, so notice how the DMD')
        print('reconstruction at rank = 3 is pretty much identical to the rank = 2 surface.\n')

        # testing function from book
        im = 0+1j

        def sech(x):
            return 1/np.cosh(x)

        def f(x, t):
            return sech(x + 3)*exp(2.3*im*t) + 2*sech(x)*np.tanh(x)*exp(im*2.8*t)

        points = 100
        x = np.linspace(-10, 10, points)
        t = np.linspace(0, 4*np.pi, points)

        # test decomposition of the function given above
        F = np.zeros((np.size(x), np.size(t)), dtype=np.complex_)
        for i, x_val in enumerate(x):
            for j, t_val in enumerate(t):
                F[i, j] = f(x_val, t_val)
        results1 = dmd.decomp(F, t, verbose=False, num_svd=1, svd_cut=True)
        results2 = dmd.decomp(F, t, verbose=False, num_svd=2, svd_cut=True)
        results3 = dmd.decomp(F, t, verbose=False, num_svd=3, svd_cut=True)

        # plotting

        # make the figure
        fig = plt.figure(figsize=(10, 7))
        surf_real_ax = fig.add_subplot(2, 2, 1, projection='3d')
        surf1_ax = fig.add_subplot(2, 2, 2, projection='3d')
        surf2_ax = fig.add_subplot(2, 2, 3, projection='3d')
        surf3_ax = fig.add_subplot(2, 2, 4, projection='3d')

        surf_real_ax = visualize.surface_data(
            np.real(F), t, x, provide_axis=True, axis=surf_real_ax)
        surf1_ax = visualize.surface_data(
            np.real(results1.Xdmd), t, x, provide_axis=True, axis=surf1_ax)
        surf2_ax = visualize.surface_data(
            np.real(results2.Xdmd), t, x, provide_axis=True, axis=surf2_ax)
        surf3_ax = visualize.surface_data(
            np.real(results3.Xdmd), t, x, provide_axis=True, axis=surf3_ax)

        surf_real_ax.set_xlabel('t')
        surf_real_ax.set_ylabel('x')
        surf_real_ax.set_zlabel('f(x,t)')
        surf_real_ax.set_title('Original function')

        surf1_ax.set_xlabel('t')
        surf1_ax.set_ylabel('x')
        surf1_ax.set_zlabel('f(x,t)')
        surf1_ax.set_title('1 Singular Value')

        surf2_ax.set_xlabel('t')
        surf2_ax.set_ylabel('x')
        surf2_ax.set_zlabel('f(x,t)')
        surf2_ax.set_title('2 Singular Values')

        surf3_ax.set_xlabel('t')
        surf3_ax.set_ylabel('x')
        surf3_ax.set_zlabel('f(x,t)')
        surf3_ax.set_title('3 Singular Values')

        # now make a plot for the normal mode analysis

        # make the figure
        fig_2 = plt.figure(figsize=(10, 7))
        time_1_ax = fig_2.add_subplot(3, 2, 1)
        time_2_ax = fig_2.add_subplot(3, 2, 3)
        time_3_ax = fig_2.add_subplot(3, 2, 5)
        space_1_ax = fig_2.add_subplot(3, 2, 2)
        space_2_ax = fig_2.add_subplot(3, 2, 4)
        space_3_ax = fig_2.add_subplot(3, 2, 6)

        space_1_ax.plot(x, np.real(results3.phi.T[0]))
        space_2_ax.plot(x, np.real(results3.phi.T[1]))
        space_3_ax.plot(x, np.real(results3.phi.T[2]))

        time_1_ax.plot(x, np.real(results3.dynamics[0]))
        time_2_ax.plot(x, np.real(results3.dynamics[1]))
        time_3_ax.plot(x, np.real(results3.dynamics[2]))

        time_1_ax.set_xlabel('t')
        time_1_ax.set_ylabel('f')
        time_1_ax.set_title('First Time Mode')

        time_2_ax.set_xlabel('t')
        time_2_ax.set_ylabel('f')
        time_2_ax.set_title('Second Time Mode')

        time_3_ax.set_xlabel('t')
        time_3_ax.set_ylabel('f')
        time_3_ax.set_title('Third Time Mode')

        space_1_ax.set_xlabel('x')
        space_1_ax.set_ylabel('f')
        space_1_ax.set_title('First Spacial Mode')

        space_2_ax.set_xlabel('x')
        space_2_ax.set_ylabel('f')
        space_2_ax.set_title('Second Spacial Mode')

        space_3_ax.set_xlabel('x')
        space_3_ax.set_ylabel('f')
        space_3_ax.set_title('Third Spacial Mode')

        fig_2.tight_layout()

        plt.show()

        return fig, fig_2

    def energy():
        '''
        need an example to show some dmd calculations envolving energy markets
        of some kind here
        '''
        True

    def kutz_mr(animate=False, plot_modes=False):
        '''
        An example taken from Kutz's book in order to show how to use
        multi-resolution DMD
        '''

        # first make the movie that has three different spacial-temporal modes

        # constants
        # large = False
        large = True
        if large:
            nx = 80
            ny = 80
            start1 = 40
            end1 = 10
            end2 = 20
            T = 10
            dt = 0.01
            time = np.linspace(0, T, T/dt+1)
            sig = 0
        else:
            nx = 8
            ny = 8
            start1 = 4
            end1 = 1
            end2 = 2
            T = 10
            dt = 1
            time = np.linspace(0, T, T/dt+1)
            sig = 0

        # make the three modes
        class mode1:
            def __init__(self):
                Xgrid, Ygrid = np.meshgrid(np.arange(nx) + 1, np.arange(ny) + 1)
                self.u = np.exp(-((Xgrid - 40)**2 / 250 + (Ygrid-40)**2 / 250))
                self.f = 5.55
                self.A = 1
                self.lam = np.exp(1j * self.f * 2 * np.pi)
                self.range = [0, 5]

        class mode2:
            def __init__(self):
                self.u = np.zeros((nx, ny), dtype=np.complex_)
                self.u[nx - start1:nx - end1, ny - start1:ny - end1] = 1
                self.f = 0.9
                self.A = 1
                self.lam = np.exp(1j * self.f * 2 * np.pi)
                self.range = [3, 7]

        class mode3:
            def __init__(self):
                self.u = np.zeros((nx, ny), dtype=np.complex_)
                self.u[0:nx - end2, 0:ny - end2] = 1
                self.f = 0.15
                self.A = 0.5
                self.lam = np.exp(1j * self.f * 2 * np.pi)
                self.range = [0, T]

        # initalize the modes
        modes = [mode1(), mode2(), mode3()]

        # make the movie
        Xclean = np.zeros((np.size(time), nx, ny))
        for ind, t in enumerate(time):
            snap = np.zeros((nx, ny))
            for mode in modes:
                if t >= mode.range[0] and t <= mode.range[1]:
                    snap = snap + np.real(mode.A*mode.u*np.real(mode.lam**t))
            Xclean[ind] = snap
        Xclean = manipulate.three_2_two(Xclean)

        # add some noise
        Xclean = Xclean + sig * \
            np.random.randn(np.shape(Xclean)[0], np.shape(Xclean)[1])

        # do the regular dmd reconstruction
        dmd_result = dmd.decomp(Xclean, time, svd_cut=False, num_svd=10)
        dmd_reg = dmd_result.Xdmd.real

        # perform the mr_dmd on the movie data
        mr_dmd = MrDMD(svd_rank=10, max_level=6, max_cycles=2)  # noqa:
        mr_dmd.fit(X=Xclean)
        dmd_data = mr_dmd.reconstructed_data.real

        # animate the sample movie
        animate = True
        plot_modes = False
        if animate:
            # reshape
            Xclean = manipulate.two_2_three(Xclean, nx)
            dmd_data = manipulate.two_2_three(dmd_data, nx)
            dmd_reg = manipulate.two_2_three(dmd_reg, nx)
            fig = plt.figure(figsize=(12, 8))
            ax_1 = fig.add_subplot(1, 3, 1)
            ax_2 = fig.add_subplot(1, 3, 2)
            ax_3 = fig.add_subplot(1, 3, 3)
            im_1 = ax_1.imshow(Xclean[0].real, interpolation='bilinear', cmap=cm.jet)
            im_2 = ax_2.imshow(dmd_data[0].real, interpolation='bilinear', cmap=cm.jet)
            im_3 = ax_3.imshow(dmd_reg[0].real, interpolation='bilinear', cmap=cm.jet)
            ax_1.axis("off")
            ax_2.axis("off")
            ax_3.axis("off")
            ax_1.set_title('Real Data')
            ax_2.set_title('Multi-Resolution DMD Reconstruction')
            ax_3.set_title('DMD Reconstruction')

            def update(t):
                price = Xclean[t].real
                im_1.set_array(price)
                price = dmd_data[t].real
                im_2.set_array(price)
                price = dmd_reg[t].real
                im_3.set_array(price)
            FuncAnimation(fig, update, interval=5, frames=np.arange(np.size(time)))
            plt.show()

        if plot_modes:
            fig = plt.figure(figsize=(10, 7))
            ax_1 = fig.add_subplot(2, 3, 1)
            ax_2 = fig.add_subplot(2, 3, 2)
            ax_3 = fig.add_subplot(2, 3, 3)
            im_1 = ax_1.imshow(np.array(modes[0].u.real, dtype=float),
                               interpolation='bilinear', cmap=cm.jet)
            im_2 = ax_2.imshow(np.array(modes[1].u.real, dtype=float),
                               interpolation='bilinear', cmap=cm.jet)
            im_3 = ax_3.imshow(np.array(modes[2].u.real, dtype=float),
                               interpolation='bilinear', cmap=cm.jet)
            ax_1.axis("off")
            ax_2.axis("off")
            ax_3.axis("off")
            ax_4 = fig.add_subplot(2, 3, 4)
            ax_5 = fig.add_subplot(2, 3, 5)
            ax_6 = fig.add_subplot(2, 3, 6)
            ax_4.plot(time, np.real(modes[0].A*modes[0].lam**time))
            ax_5.plot(time, np.real(modes[1].A*modes[1].lam**time))
            ax_6.plot(time, np.real(modes[2].A*modes[2].lam**time))
            plt.show()


class energy:

    '''
    This class will hold all of the necessary function for manipulating
    the energy price data in this project along with the DMD results
    '''

    def imp_prices(name, start=1):
        '''
        This is a simple function that will import price data as a
        matrix in numpy and return the resulting matrix.

        inputs:
        name - string with name of the csv file

        outputs:
        X - numpy array with price data over time
        '''

        X = pd.read_csv(name, header=None)
        X = X.values

        return X

    def imp_locations(name):
        '''
        This is a simple function that will import location data as a
        matrix in numpy and return the resulting matrix.

        inputs:
        name - string with name of the csv filei
        outputs:
        numpy array with price data over time
        '''

        X = np.genfromtxt(name, delimiter=',')
        X = X[1:]
        return X

    def data_wrangle(data, start_loc, end_loc, start_time, end_time):
        '''
        This function will just return a numpy array with the given
        specifications of where you want to start the time and the
        location examples.

        inputs:
        data - price matrix
        start_loc - first location index
        end_loc - last location index
        start_time - first time step
        end_time - last time step

        output:
        X - matrix of manipulated price
        '''

        X = data[start_loc:end_loc]
        X = X.T
        X = X[start_time:end_time]
        X = X.T

        return X

    def data_wrangle_vec(data, loc_vec, start_time, end_time):
        '''
        This function will just return a numpy array with the given
        specifications of where you want to start the time and the
        location vector.

        inputs:
        data - price matrix
        loc_vec - list of location indicies that will be used
        start_time - first time step
        end_time - last time step

        output:
        X - matrix of manipulated price
        '''

        X = np.take(data, loc_vec)
        X = X.T
        X = X[start_time:end_time]
        X = X.T

        return X

    def calc_end_error(end_val, error, data, verbose=False):
        '''
        This function calculates the 2-norm error of a matrix of
        residual values based on the last "end_val" times.
        The input is a list of residual matricies like the
        "sv_dependence" function returns.

        inputs:
        end_val - the last "N" number of time steps to calulate error on
        error - matrix of error measurements
        data - original data input into dmd
        verbose - verbose argument

        output:
        end_error - % error of the last "end_val" time steps
        '''

        # determine the 2-norm of the data in the last time measurements
        time_len = np.size(data[0])
        data = data.T
        data = data[time_len - end_val:]
        data = data.T
        data_norm = la.norm(data)

        # initalize a list for the error values
        end_error = []

        # loop through to find the error for each sv
        for test_ind, res_matrix in enumerate(error):

            # grab the last end_vals
            res_matrix = res_matrix.T
            res_matrix = res_matrix[time_len - end_val:]
            res_matrix = res_matrix.T

            # calculate the error
            error = la.norm(res_matrix) / data_norm * 100

            # append the error
            end_error.append(error)

            if verbose:
                print('------------------------------------')
                print('Test #'+str(test_ind))
                print()
                print(res_matrix)
                print(la.norm(res_matrix))
                print()
                print(data)
                print(data_norm)
                print('Error:', error)
                print('------------------------------------')
                print()

        return end_error

    def calc_opt_sv_cut(results, data, N=24, verbose=False):
        '''
        From a singular value sensitivity test class, this will calculate
        the optimal rank reduction given a time period to test on.

        inputs:
        results - class returned from sv_sensitivity
        N - time period on which to test

        outputs:
        opt_sv - optimal singular value to cut on
        end_error - array that has the error for each rank reduction
        '''

        # determine the optimal number of singular values
        end_error = energy.calc_end_error(N, results.res_matrix, data, verbose=False)
        opt_sv = end_error.index(min(end_error)) + 1
        if verbose:
            print('For a time period of', N, 'hours...')
            print('\nOptimal Singular Value Reduction Identified:', opt_sv)
            print('Percentage:', opt_sv/np.size(results.num_svd)*100, '%')

        return opt_sv, end_error


class dmd:

    '''
    This class contains the functions needed for performing a full DMD
    on any given matrix. Depending on functions being used, different
    outputs can be achived.
    This class also contains functions useful to the analysis of DMD
    results and intermediates.
    '''

    def decomp(Xf, time, verbose=False, rank_cut=True, esp=1e-2, svd_cut=False,
               num_svd=1, do_SVD=True, given_svd=False):
        '''
        This function performs the basic DMD on a given matrix A.
        The general outline of the algorithm is as follows...
        1)  Break up the input X matrix into time series for 1 to n-1 (X)
            and 2 to n (X) where n is the number of time intervals (X_p)
            (columns). This uses the manipulate class's function "split".
        2)  Compute the Singular Value Decomposition of X. X = (U)(S)(Vh)
        3)  Compute the A_t matrix. This is related to the A matrix which
            gives A = X * Xp. However, the At matrix has been projected
            onto the POD modes of X. Therefore At = U'*A*U. (' denomates
            a matrix transpose)
        4)  Compute the eigendecomposition of the At matrix. At*W=W*L
        5)  Compute the DMD modes of A by SVD reconstruction. Finally, the
            DMD modes are given by the columns of Phi.
            Phi = (Xp)(V)(S^-1)(W)
        6)  Compute the discrete and continuous time eigenvalues
            lam (discrete) is the diagonal matrix of eigenvalues of At.
            omg (continuous) = ln(lam)/dt
        7) 	Compute the amplitude of each DMD mode (b). This is a vector
            which applies to this system: Phi(b)=X_1 Where X_1 is the first
            column of the input vector X. This requires a linear equation
            solver via scipy.
        8)  Reconstruct the matrix X from the DMD modes (Xdmd).

        inputs:
        X - (mxn) Spacial Temporal Matrix
        time - (nx1) Time vector

        outputs:
        (1) Phi - DMD modes
        (2) omg - discrete time eigenvalues
        (3) lam - continuous time eigenvalues
        (4) b - amplitudes of DMD modes
        (5) Xdmd - reconstructed X matrix from DMD modes
        (6) rank - the rank used in calculations

        *** all contained in a class ***
        ***  see ### (10) ### below  ***

        options:
        verbose - boolean for more information
        svd_cut - boolean for truncation of SVD values of X
        esp - value to truncate singular values lower than
        rank_cut - truncate the SVD of X to the rank of X
        num_svd - number of singular values to use
        do_SVD - tells the program if the svd is provided to it or not
        '''

        if verbose:
            print('Entering Dynamic Mode Decomposition:\n')

        # --- (1) --- #
        # split the Xf matrix
        X, Xp = manipulate.split(Xf)
        if verbose:
            print('X = \n', X, '\n')
            print('X` = \n', Xp, '\n')

        ### (2) ###  # noqa:
        # perform a singular value decompostion on X
        if do_SVD:
            if verbose:
                'Performing singular value decompostion...\n'
            U, S, Vh = la.svd(X)
        else:
            if verbose:
                'Singular value decompostion provided...\n'
            U, S, Vh = given_svd

        if verbose:
            print('Singular value decomposition:')
            print('U: \n', U)
            print('S: \n', S)
            print('Vh: \n', Vh)
            print('Reconstruction:')
            S_m = np.zeros(np.shape(X))
            for i in range(len(list(S))):
                S_m[i, i] = S[i]
            recon = np.dot(np.dot(U, S_m), Vh)
            print('X =\n', recon)

        # perfom desired truncations of X
        if svd_cut:
            rank_cut = False
        if rank_cut:  # this is the default truncation
            rank = 0
            for i in S:
                if i > esp:
                    rank += 1
            if verbose:
                print('Singular Values of X:', '\n', S, '\n')
                print('Reducing Rank of System...\n')
            Ur = U[:, 0:rank]
            Sr = S[0:rank]
            Vhr = Vh[0:rank, :]
            if verbose:
                recon = np.dot(np.dot(Ur, np.diag(Sr)), Vhr)
                print('Rank Reduced reconstruction:\n', 'X =\n', recon)
        elif svd_cut:
            rank = num_svd
            if verbose:
                print('Singular Values of X:', '\n', S, '\n')
                print('Reducing Rank of System to n =', num_svd, '...\n')
            Ur = U[:, 0:rank]
            Sr = S[0:rank]
            Vhr = Vh[0:rank, :]
            if verbose:
                recon = np.dot(np.dot(Ur, np.diag(Sr)), Vhr)
                print('Rank Reduced reconstruction:\n', 'X =\n', recon)

        # return the condition number to view singularity
        condition = max(Sr) / min(Sr)
        smallest_svd = min(Sr)
        svd_used = np.size(Sr)
        if verbose:
            condition = max(Sr) / min(Sr)
            print('Condition of Rank Converted Matrix X:', '\nK =', condition, '\n')

        # make the singular values a matrix and take the inverse
        Sr_inv = np.diag([i ** -1 for i in Sr])
        Sr = np.diag(Sr)

        ### (3) ###  # noqa:
        # now compute the A_t matrix
        Vr = Vhr.conj().T
        At = Ur.conj().T.dot(Xp)
        At = At.dot(Vr)
        At = At.dot(la.inv(Sr))
        if verbose:
            print('A~ = \n', At, '\n')

        ### (4) ###  # noqa:
        # perform the eigen decomposition of At
        L, W = la.eig(At)
        # also determine the number of positive eigenvalues
        pos_eigs = np.count_nonzero((L > 0))

        ### (5) ###  # noqa:
        # compute the DMD modes
        # phi = Xp @ Vhr.conj().T @ Sr_inv @ W
        phi = np.dot(Xp, Vhr.conj().T)
        phi = np.dot(phi, Sr_inv)
        phi = np.dot(phi, W)

        if verbose:
            print('DMD Mode Matrix:', '\nPhi =\n', phi, '\n')

        ### (6) ###   # noqa:
        # compute the continuous and discrete eigenvalues
        dt = time[1] - time[0]
        lam = L
        omg = np.log(lam) / dt
        if verbose:
            print('Discrete time eigenvalues:\n', 'Lambda =', L, '\n')
            print('Continuous time eigenvalues:\n', 'Omega =', np.log(L) / dt, '\n')
            print('Number of positive eigenvalues: ', pos_eigs, '\n')

        ### (7) ###  # noqa:
        # compute the amplitude vector b by solving the linear system described.
        # note that a least squares solver has to be used in order to approximate
        # the solution to the overdefined problem
        x1 = X[:, 0]
        b = la.lstsq(phi, x1)
        b = b[0]
        if verbose:
            print('b =\n', b, '\n')

        ### (8) ###  # noqa:
        # finally reconstruct the data matrix from the DMD modes
        length = np.size(time)  # number of time measurements
        # initialize the time dynamics
        dynamics = np.zeros((rank, length), dtype=np.complex_)
        for t in range(length):
            omg_p = np.array([exp(i * time[t]) for i in omg])
            dynamics[:, t] = b * omg_p

        if verbose:
            print('Time dynamics:\n', dynamics, '\n')

        # reconstruct the data
        Xdmd = np.dot(phi, dynamics)
        if verbose:
            print('Reconstruction:\n', np.real(Xdmd), '\n')
            print('Original:\n', np.real(Xf), '\n')

        ### (9) ###  # noqa:
        # calculate some residual value
        res = np.real(Xf - Xdmd)
        error = la.norm(res) / la.norm(Xf)
        if verbose:
            print('Reconstruction Error:', round(error * 100, 2), '%')

        ### (10) ###  # noqa:
        # returns a class with all of the results
        class results():
            def __init__(self):
                self.phi = phi
                self.omg = omg
                self.lam = lam
                self.b = b
                self.Xdmd = Xdmd
                self.error = error * 100
                self.rank = rank
                self.svd_used = svd_used
                self.condition = condition
                self.smallest_svd = smallest_svd
                self.pos_eigs = pos_eigs
                self.dynamics = dynamics
                self.svd_used = svd_used

        return results()

    def predict(dmd, t):
        '''
        This function will take a DMD decomposition output
        result and a desired time incremint prediction and
        produce a prediction of the system at the given time.

        inputs:
        dmd - class that comes from the function "decomp"
        t - future time for prediction

        outputs:
        x - prediction vector (real part only)
        '''

        # finally reconstruct the data matrix from the DMD modes
        dynamics = np.zeros((dmd.rank, 1), dtype=np.complex_)
        omg_p = np.array([exp(i * t) for i in dmd.omg])
        dynamics = dmd.b * omg_p
        x = np.real(np.dot(dmd.phi, dynamics))

        return x

    def dmd_specific_svd(Xf):
        '''
        This is a helper function which will split the data and
        perform a singular value decomposition based on whatever the
        input data is and return the outputs for scipy.
        '''

        X, Xp = manipulate.split(Xf)
        result = la.svd(X)

        return result

    def mode_analysis(data, dmd_results, N=np.arange(2), analyze=False, plot=True):
        '''
        This function will take the time dynamics and spacial dynamics and show
        a plot for the number of modes that have been specified.

        inputs:
        results - results class from dmd.decomp
        data - data used in the decomposition
        N - number of modes that you want to plot

        outputs:
        results - class with useful information
        fig - figure of the modes
        '''

        # make the time and space vectors
        time = np.arange(np.shape(data)[1])
        space = np.arange(np.shape(data)[0])

        # check feasibility
        if np.size(N) > dmd_results.svd_used:
            print('Too many singular values requested!')
            print('Reducing analysis to N =', dmd_results.svd_used)
            N = np.arange(dmd_results.svd_used)

        # do an analysis of the modes if ased for (default yes)
        if analyze:
            results = []
            True

        # make a plot if ased for (default no)
        if plot:

            # create the figure and the axes
            fig = plt.figure(figsize=(10, 2.3 * np.size(N)))
            time_axes = [True for i in N]
            space_axes = [True for i in N]

            # through the number of modes that are desired to be analyzed
            for ind, n in enumerate(N):

                # set up each axis
                time_axes[ind] = fig.add_subplot(np.size(N), 2, ind*2 + 1)
                time_axes[ind].set_xlabel('Time')
                time_axes[ind].set_ylabel('Price ($)')
                title = 'Time Mode #'+str(n+1)+' || eig = ' + \
                    str(round(dmd_results.omg[n], 2))
                time_axes[ind].set_title(title)
                time_axes[ind].plot(time, np.real(dmd_results.dynamics[n]))

                space_axes[ind] = fig.add_subplot(np.size(N), 2, ind*2 + 2)
                space_axes[ind].set_xlabel('Location Index')
                space_axes[ind].set_ylabel('Price ($)')
                space_axes[ind].set_title('Spacial Mode #'+str(n+1))
                space_axes[ind].plot(space, np.real(dmd_results.phi.T[n]))
            fig.tight_layout()

        # return the desired stuff
        if plot and analyze:
            return fig and results
        elif plot and not analyze:
            return fig
        elif analyze and not plot:
            return results


class big_data:

    '''
    This class holds functions that are needed to deal with the 1_hour price data.
    '''

    def make_matrix(folder, file_name, dest):
        """
        Based on a folder path, this function will load all of the data in the folder
        and make a unified matrix of price data. This matrix will be made into a csv
        file and dumped into the destination path provided.

        : param folder: str of the folder path
        : param file_name: str of the file name
        : param dest: str of the path destination of the data
        """

        # initialize the matrix
        mat = np.array([[]])
        min_col = 1e20

        # get the file names
        rows = os.listdir(folder)
        num_files = len(rows)

        # get the first value in the matrix
        name = folder+rows[0]
        X = np.genfromtxt(name, delimiter=',')
        if X.size < min_col:
            min_col = X.size
        mat = np.array([X])

        # make a names array for the bad nodes
        bad_nodes = []
        good_nodes = []

        # loop through the rest of the files
        num = 0
        for row in rows[1:]:
            name = folder+row
            X = np.genfromtxt(name, delimiter=',')

            if np.all(X) and (X.min() > 0) and (X.max() < 200):
                X = np.array([X])
                mat = np.concatenate((mat, X))
                good_nodes.append(row)
            else:
                bad_nodes.append(row)
                print(row, 'is a bad node.')

            num += 1
            print(num, 'out of', num_files)

        bad_nodes = np.array(bad_nodes)
        good_nodes = np.array(good_nodes)

        # write the matricies as a csv file to its destination
        np.savetxt(dest+file_name+'.csv', mat, delimiter=',', fmt='%.4f')
        np.savetxt(dest+'bad_nodes_mod'+'.csv', bad_nodes, delimiter='\n', fmt="%s")
        np.savetxt(dest+'good_nodes_mod'+'.csv',
                   good_nodes, delimiter='\n', fmt="%s")


class augment:

    '''
    This class hold the information on how to perform augmented DMD.
    '''

    def augment_matrix(x, n):
        '''
        Function to take a vector x and returns an augmented matrix X which
        has n rows.
        '''

        # length of full time series
        num_elements = x.shape[0]

        # length of each row in the X matrix
        len_row = num_elements - n + 1

        # initalize the matrix
        X = []

        # loop over each row
        for row_num in range(n):

            # grab the smaller vector
            small_vec = x[row_num:row_num + len_row]

            # append the vector
            X.append(small_vec)

        return np.array(X)

    def make_forecast(data, train_start, train_end, num_predict=48, rank=8, verbose=False,
                      give_recon=False):
        '''
        This function will make a 48 hour forecast using augmented DMD.
        '''

        # get the time measurements desired
        data = data.T[train_start:train_end].T
        if verbose:
            print('start:', train_start)
            print('end:', train_end)

        # loop through each row of data and make a forecast
        forecast = []
        error_vec = []
        for x in data:

            # determine how many rows and the length of each row
            num_rows = int((train_end - train_start) / 2)
            if verbose:
                print('Rows:', num_rows)

            # make the augmented matrix
            X = augment.augment_matrix(x, num_rows)
            # DMD
            time = np.arange(X[0].shape[0])
            if verbose:
                print('augmented shape:', X.shape)
            dmd_results = dmd.decomp(X, time, svd_cut=True, num_svd=rank)

            # predict the future measurements
            x_dmd_future = []
            time_future = np.arange(num_predict) + x.shape[0]
            for t in time_future:
                x_dmd_future.append(dmd.predict(dmd_results, t)[0])
            x_dmd_future = np.array(x_dmd_future)
            forecast.append(x_dmd_future)
            error_vec.append(dmd_results.error)

        # return the forecast
        if give_recon:
            return np.array(forecast), error_vec
        else:
            return np.array(forecast)


class visualize:

    '''
    This class holds all of the functions needed for visualizing of DMD
    results and the input data into DMD.
    '''

    def surface_data(F, x, t, bounds_on=False, provide_axis=False, axis=False,
                     bounds=[[0, 1], [0, 1], [0, 1]]):
        '''
        This function will create a surface plot of given a set of data
        for f(x),x,t. f(x) must be given in matrix format with evenly
        spaced x and t corresponding the A matrix.

        inputs:
        f - spacial-temporal data
        x - spacial vector
        t - time vector

        outputs:
        surf - object of the 3D plot

        options:
        bounds_on - boolean to indicate bounds wanted
        bounds - Optional array that contains the bounds desired to put on
                 the axes. Sample input: [[0,1],[0,1],[0,1]] for f(x),x,t.
        '''

        # first make a meshgrid with the t and x vector.
        # we first define the x values as the rows and t as the columns
        # in order to be consistent with general DMD structure.
        X, T = np.meshgrid(x, t)

        # Create 3D figure if you are not providing the axis
        if provide_axis:
            ax = axis
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        # Plot f(x)
        ax.plot_surface(X, T, F, linewidth=0, cmap=cm.coolwarm, antialiased=True)

        # give the two options for what to provide
        if provide_axis:
            return ax
        else:
            return ax, fig

    def surface_function(f, x, t, bounds_on=False, bounds=[[0, 1], [0, 1], [0, 1]]):
        '''
        This function will create a surface plot of given a set of data
        for f(x),x,t.

        inputs:
        f - input function
        x - spacial vector
        t - time vector

        outputs:
        surf - object for the figure that is created

        options:
        bounds_on - boolean to indicate bounds wanted
        bounds - Optional array that contains the bounds desired to put on
                 the axes. Sample input: [[0,1],[0,1],[0,1]] for f(x),x,t.
        '''

        # first make a meshgrid with the t and x vector.
        # we first define the x values as the rows and t as the columns
        # in order to be consistent with general DMD structure.
        x_len = np.size(x)
        t_len = np.size(t)
        X, T = np.meshgrid(x, t)

        # now evaluate the function.
        F = np.zeros((t_len, x_len))
        for i, x_val in enumerate(x):
            for j, t_val in enumerate(t):
                F[j, i] = f(x_val, t_val)

        # Create 3D figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot f(x)
        surf = ax.plot_surface(X, T, F, linewidth=0,
                               cmap=cm.coolwarm, antialiased=True)

        return surf


class cal_plot:

    '''
    This class will hold all necessary function for the plotting
    of any energy stuff on a California map
    '''

    def plot_price(data, locations, time=0):
        '''
        This will be the basic plotting function to allow you to plot
        energy prices on a colormap on california.

        Inputs:
        data - (mxn) matrix where columns are times and row are locations
        locations - standard
        time - integer for time you want to plot
        '''

        fig, ax = plt.subplots()
        lat = locations[:, 0]
        lon = locations[:, 1]
        price = data[:, time]

        max_price = 70
        min_price = 20

        data_plot = ax.scatter(lon, lat, alpha=0.5, c=price,
                               cmap=cm.jet, s=8, vmin=min_price, vmax=max_price)
        plt.colorbar(data_plot, label='Price ($)')
        ax.axis([-126.30, -104, 32.45, 46.73])
        map_img = mpimg.imread('map.png')
        plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
        ax.axis("off")
        title = 'Time: '+str(time)
        ax.set_title(title)

        return fig

    def plot_single(price, locations, bar_min, bar_max, cb_label='Price ($)'):
        '''
        This will be the basic plotting function to allow you to plot
        energy prices on a colormap on california.

        Inputs:
        data - (mxn) matrix where columns are times and row are locations
        locations - standard
        time - integer for time you want to plot
        '''

        fig, ax = plt.subplots()
        lat = locations[:, 0]
        lon = locations[:, 1]

        data_plot = ax.scatter(lon, lat, alpha=0.5, c=price,
                               cmap=cm.jet, s=8, vmin=bar_min, vmax=bar_max)
        plt.colorbar(data_plot, label=cb_label)
        ax.axis([-126.30, -104, 32.45, 46.73])
        map_img = mpimg.imread('map.png')
        plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
        ax.axis("off")

        return fig, ax

    def plot_cluster(groups, locations):
        '''
        This will be the basic plotting function to allow you to plot
        energy prices on a colormap on california.

        Inputs:
        data - (mxn) matrix where columns are times and row are locations
        locations - standard
        time - integer for time you want to plot
        '''

        fig, ax = plt.subplots()
        lat = locations[:, 0]
        lon = locations[:, 1]

        data_plot = ax.scatter(lon, lat, alpha=0.5, c=groups,  # noqa:
                               cmap=cm.tab10, s=8, vmin=0, vmax=10)
        ax.axis([-126.30, -104, 32.45, 46.73])
        map_img = mpimg.imread('map.png')
        plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
        ax.set_title('Cluster Analysis')
        ax.axis("off")

        return fig, ax

    def animate_price(data, locations, time=100, speed=150, label='Price'):
        '''
        This will be the basic plotting function to allow you to plot
        energy prices on a colormap on california.

        Inputs:
        data - (mxn) matrix where columns are times and row are locations
        locations - standard
        time - integer for time you want to plot
        speed - input into the animation object for frame update speed
        '''

        fig, ax = plt.subplots()
        lat = locations[:, 0]
        lon = locations[:, 1]
        price = data[:, 0]

        max_price = 50
        min_price = 20

        data_plot = ax.scatter(lon, lat, alpha=0.5, c=price,
                               cmap=cm.jet, s=8, vmin=min_price, vmax=max_price)
        plt.colorbar(data_plot, label='Price ($)')
        ax.axis([-126.30, -104, 32.45, 46.73])
        map_img = mpimg.imread('map.png')
        plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
        ax.axis("off")
        label = label + '\n'

        def update(t):
            price = data[:, t]
            title = 'Time: '+str(t)
            ax.set_title(label+title)
            data_plot.set_array(price)

        animation = FuncAnimation(fig, update, interval=speed, frames=time)

        return animation

    def animate_error(error, locations, time=100, speed=150):
        '''
        This will be the basic plotting function to allow you to plot
        energy prices on a colormap on california.

        Inputs:
        data - (mxn) matrix where columns are times and row are locations (0,1)
        locations - standard
        time - integer for time you want to plot
        speed - input into the animation object for frame update speed
        '''

        fig, ax = plt.subplots()
        lat = locations[:, 0]
        lon = locations[:, 1]
        price = error[:, 0]

        max_error = 60
        min_error = 10

        data_plot = ax.scatter(lon, lat, alpha=0.5, c=price,
                               cmap=cm.jet, s=8, vmin=min_error, vmax=max_error)
        plt.colorbar(data_plot, label='error')
        ax.axis([-126.30, -104, 32.45, 46.73])
        map_img = mpimg.imread('map.png')
        plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
        ax.axis("off")

        def update(t):
            price = error[:, t]
            title = 'Time: '+str(t)
            ax.set_title(title)
            data_plot.set_array(price)

        animation = FuncAnimation(fig, update, interval=speed, frames=time)  # noqa:
        plt.show()


class basic_plots:

    '''
    This will hold functions that are used for basic plotting of energy prices
    or related information
    '''

    def plot_energy(data, indecies, start, num_vals):
        '''
        This function will plot the energy prices for a given vector of
        indicies corresponding to location in a data matrix. The number of
        time measurements and where to start can also be specified.

        inputs:
        data - energy price data
        indecies - array of integers corresponding to locations
        start - time measurement to start at
        num_vals - number of time measurements to plot

        output:
        fig - figure containing the plot
        '''

        fig, ax = plt.subplots()
        indecies = list(indecies)
        for i in indecies:
            ax.plot(data[i, start:start + num_vals])
        title = 'Energy Price Visualization\n'
        title = title + str(len(indecies)) + ' Locations Shown'
        ax.set(xlabel='time (days)', ylabel='price ($)',
               title='Energy Price Visualization')
        return fig


if __name__ == '__main__':

    print(examples)
