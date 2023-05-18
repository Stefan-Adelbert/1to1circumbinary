import rebound
import numpy as np
from timeit import default_timer as timer
import os
import datetime
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import pickle


class Counter(object):
    """Counter which keeps track of how many simulation packages are done already when multiprocessing is in use."""
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1
            return self.val.value

    @property
    def value(self):
        return self.val.value


counter = Counter(0)
start0 = timer()
output = "./output.txt"

def run_sim(mode, path, filename, a_p, e_p=0.3, mass_p=1e-4, mu_p=0.5, phi_p=np.pi, per=np.pi / 2, inc=0, Omega=0,
            e_bin=0.04, mu_bin=0.4, phi_bin=0, n_orbits=1e5, dt=10, export=None):
    """Run a single Rebound simulation
    
    Arguments:
    mode -- "1P" for single planet simulations; "2P" for planets in 1:1 resonance; "{x}:{y}" for planets in x:y resonance
    path -- Path at which to store output
    filename -- Name for output file
    a_p -- planetary semi-major axis
    e_p -- planetary eccentricity (default 0.3)
    mass_p -- ratio of planet mass to binary mass ((m_1+m_2)/(M_1+M_2)) (default 1e-4)
    mu_p -- planetary mass ratio (m_2/(m_1+m_2)) (default 0.5)
    phi_p -- initial true anomaly of the second planet (default pi)
    per -- initial argument of pericenter (small omega) (default pi/2)
    inc -- planetary inclination (default 0)
    Omega -- planetary longitude of ascending node (default 0)
    e_bin -- binary eccentricity (default 0.04)
    mu_bin -- binary mass ratio (M_2/(M_1+M_2)) (default (0.4))
    phi_bin -- initial true anomaly of binary (default 0)
    n_orbits -- Number of binary orbits to be simulated (default 1e5)
    dt -- time step at which termination condition is checked and (if desired) output created (default 10)
    export -- determines output format ("txt", "sa", "pickle", ...) (default None)

    Returns:
    survived_orbits -- number of binary orbits until simulation terminated
    result -- integer denoting, if the simulation remained stable (0) or was terminated earlier than n_orbits (1)
    """

    if not mode in ["1P", "2P", "2:1", "3:2", "3:1"]:
        # code could handle other ratios than 2:1, 3:2 and 3:1, but those were the only ones we used,
        # so this was an easier way of avoiding typos
        print("unknown simulation mode chosen")
        return -1

    sim = rebound.Simulation()

    # initialization of the binary
    sim.add(m=1 - mu_bin)
    sim.add(m=mu_bin, a=1, e=e_bin, f=phi_bin)

    # initialization of the planets
    sim.add(m=mass_p * (1 - mu_p), a=a_p, e=e_p, inc=inc, Omega=Omega, omega=per)
    if mode == "2P":
        sim.add(m=mass_p * mu_p, a=a_p, e=e_p, inc=inc, Omega=Omega, omega=per, f=phi_p)
    elif ":" in mode:
        split = mode.split(":")
        ratio = float(split[0])/float(split[1])
        sim.add(m=mass_p * mu_p, a=a_p * ratio ** (2 / 3), e=e_p, inc=inc, Omega=Omega, omega=per, f=phi_p)

    sim.move_to_com()

    t_bin = 2. * np.pi
    t_end = n_orbits * t_bin

    # initialization of the return parameters
    survived_orbits = n_orbits
    result = 0

    # initialization of the output file
    if export == "txt":
        f = open(path + filename + ".sim", "w")
        if mode == "2P":
            f.write("e_p={:.4f}, mass_p={:e}, mu_p={:.4f}, phi_p={:.4f}*pi, per={:.4f}*pi, "
                    "inc={:.4f}*pi, e_bin={:.4f}, mu_bin={:.4f}, phi_bin={:.4f}\n\n"
                    .format(e_p, mass_p, mu_p, phi_p / np.pi, per / np.pi, inc / np.pi,
                            e_bin, mu_bin, phi_bin / np.pi))
            f.write("t; a_1; a_2; e_1; e_2; per_1; per_2; lambda_1; lambda_2; inc_1; inc_2; "
                    "Omega_1; Omega_2; j_1; j_2\n")
        elif mode == "1P":
            f.write("e_p={:.4f}, mass_p={:e}, mu_p={:.4f}, phi_p={:.4f}*pi, per={:.4f}*pi, "
                    "inc={:.4f}*pi, e_bin={:.4f}, mu_bin={:.4f}, phi_bin={:.4f}\n\n"
                    .format(e_p, mass_p, mu_p, phi_p / np.pi, per / np.pi, inc / np.pi,
                            e_bin, mu_bin, phi_bin / np.pi))
            f.write("t; a_p; e_p; per_p; lambda_p; inc_p; Omega_p; j_p\n")
    elif export == "sa":
        sim.simulationarchive_snapshot(path + filename + ".sa", deletefile=True)
    elif export == "pickle":
        Pickle = pickle.Pickler(open(path + filename + ".pickle", "wb"))
        params = [e_p, mass_p, mu_p, phi_p, per, inc, e_bin, mu_bin, phi_bin]
        Pickle.dump(params)

    # actual integration, halting repeatedly after a time of dt (in binary periods)
    # in order to check termination condition and output data
    for t in np.linspace(0, t_end, round(n_orbits/dt)+1):

        sim.integrate(t, exact_finish_time=0)

        # data output (writing location and velocities, as other parameters can be calculated from those)
        if mode == "2P" and export == "txt":
            f.write((4*6*"{};" + "{}\n").format(sim.t / t_bin,
                            sim.particles[0].x, sim.particles[0].y, sim.particles[0].z,
                            sim.particles[0].v_x, sim.particles[0].v_y, sim.particles[0].v_z,
                            sim.particles[1].x, sim.particles[1].y, sim.particles[1].z,
                            sim.particles[1].v_x, sim.particles[1].v_y, sim.particles[1].v_z,
                            sim.particles[2].x, sim.particles[2].y, sim.particles[2].z,
                            sim.particles[2].v_x, sim.particles[2].v_y, sim.particles[2].v_z,
                            sim.particles[3].x, sim.particles[3].y, sim.particles[3].z,
                            sim.particles[3].v_x, sim.particles[3].v_y, sim.particles[3].v_z))
        elif mode == "1P" and export == "txt":
            f.write((3*6*"{};" + "{}\n").format(sim.t / t_bin,
                            sim.particles[0].x, sim.particles[0].y, sim.particles[0].z,
                            sim.particles[0].v_x, sim.particles[0].v_y, sim.particles[0].v_z,
                            sim.particles[1].x, sim.particles[1].y, sim.particles[1].z,
                            sim.particles[1].v_x, sim.particles[1].v_y, sim.particles[1].v_z,
                            sim.particles[2].x, sim.particles[2].y, sim.particles[2].z,
                            sim.particles[2].v_x, sim.particles[2].v_y, sim.particles[2].v_z))
        elif export == "sa":
            sim.simulationarchive_snapshot(path + filename + ".sa")
        elif export == "pickle":
            part = pd.DataFrame(columns=["x", "y", "z", "v_x", "v_y", "v_z"], index=range(len(sim.particles)))
            for k in range(len(sim.particles)):
                part["x"][k] = sim.particles[k].x
                part["y"][k] = sim.particles[k].y
                part["z"][k] = sim.particles[k].z
                part["v_x"][k] = sim.particles[k].v_x
                part["v_y"][k] = sim.particles[k].v_y
                part["v_z"][k] = sim.particles[k].v_z
            Pickle.dump(part)

        # checking termination condition (deviation in semi-major axes of more than 20%)
        if mode == "2P":
            da = np.abs(sim.particles[2].a - sim.particles[3].a)
            small_a = np.amin([sim.particles[2].a, sim.particles[3].a])
        elif mode == "1P":
            da = np.abs(sim.particles[2].a - a_p)
            small_a = sim.particles[2].a
        elif ":" in mode:
            da = np.abs(sim.particles[2].a - sim.particles[3].a/ratio**(2/3))
            small_a = np.amin([sim.particles[2].a, sim.particles[3].a/ratio**(2/3)])
        if result == 0 and (small_a < 0 or da / small_a > 0.2):
            survived_orbits = sim.t / t_bin
            result = 1
            break

    # trying to close the output file; try-except shouldn't be necessary, but you never know.
    try:
        f.close()
    except:
        ""  # yes, I am aware that this is horrible.

    if export is None or export == "None":
        sim.simulationarchive_snapshot(path + filename + ".sa", deletefile=True)
        # provides point to restart simulation, if desired

    return survived_orbits, result



def run_parameter_range(parameters, path="/", outputfile="./output.txt", n_orbits=1e5, dt=10,
                        run="linear", export=None, cpus=8):
    """Runs a range of sRebound simulations specified in parameters using run_sim

    Arguments:
    parameters -- dictionary containing parameter values and ranges for the simulations
    path -- path at which to store data
    outputfile -- relative path and filename at which to store output describing how many simulations are done already
    n_orbits -- Number of binary orbits to be simulated (default 1e5)
    dt -- time step at which termination condition is checked and (if desired) output created (default 10)
    run -- String determining, if simulations are run one after the other (default "linear") or via multiprocessing ("parallel")
    export -- determines output format ("txt", "sa", "pickle", ...) for simulation results (default None)
    cpus -- number of CPUs to be used for multiprocessing (default 8)
    """

    # initialization of timer to later calculate and estimated time of completion (ETC)
    start = timer()
    global start0
    start0 = timer()
    global output
    output = outputfile

    n = 0
    N = 1
    # total number of simulations to be run
    for param in parameters:
        if param != "aq_p" and type(parameters[param]) != str:
            N = N * len(parameters[param])


    out = open(outputfile, "w")
    out.write("starting time: " + str(datetime.datetime.now()) + "\n")
    out.write("Number of calculations: " + str(N) + "\n")
    out.close()

    mode = parameters["mode"]
    tupels = []

    # brute force way to create subfolders for each varied parameter
    # would be prettier as a recursive function, but I couldn't be bothered
    # as it barely affects performance
    for phi_bin in parameters["phi_bin"]:
        path_phi_bin = path
        if len(parameters["phi_bin"]) > 1:
            path = (path + "phi_bin={:4f}pi/").format(phi_bin / np.pi)
            os.makedirs(path, exist_ok=True)
        for mu_bin in parameters["mu_bin"]:
            path_mu_bin = path
            if len(parameters["mu_bin"]) > 1:
                path = (path + "mu_bin={:4f}/").format(mu_bin)
                os.makedirs(path, exist_ok=True)
            for e_bin in parameters["e_bin"]:
                path_e_bin = path
                if len(parameters["e_bin"]) > 1:
                    path = (path + "e_bin={:4f}/").format(e_bin)
                    os.makedirs(path, exist_ok=True)
                for inc in parameters["inc"]:
                    path_inc = path
                    if len(parameters["inc"]) > 1:
                        path = (path + "inc={:.4f}pi/").format(inc / np.pi)
                        os.makedirs(path, exist_ok=True)
                    for Omega in parameters["Omega"]:
                        path_Omega = path
                        if len(parameters["Omega"]) > 1:
                            path = (path + "Omega={:.4f}pi/").format(Omega / np.pi)
                            os.makedirs(path, exist_ok=True)
                        for per in parameters["per"]:
                            path_per = path
                            if len(parameters["per"]) > 1:
                                path = (path + "per={:.4f}pi/").format(per / np.pi)
                                os.makedirs(path, exist_ok=True)
                            for phi_p in parameters["phi_p"]:
                                path_phi_p = path
                                if len(parameters["phi_p"]) > 1:
                                    path = (path + "phi_p={:.4f}pi/").format(phi_p / np.pi)
                                    os.makedirs(path, exist_ok=True)
                                for mu_p in parameters["mu_p"]:
                                    path_mu_p = path
                                    if len(parameters["mu_p"]) > 1:
                                        path = (path + "mu_p={:.4f}/").format(mu_p)
                                        os.makedirs(path, exist_ok=True)
                                    for mass_p in parameters["mass_p"]:
                                        path_mass_p = path
                                        if len(parameters["mass_p"]) > 1:
                                            path = (path + "mass_p={:e}/").format(mass_p)
                                            os.makedirs(path, exist_ok=True)
                                        for e_p in parameters["e_p"]:
                                            path_e_p = path
                                            if len(parameters["e_p"]) > 1:
                                                path = (path + "e_p={:.4f}/").format(e_p)
                                                os.makedirs(path, exist_ok=True)

                                            # simulations are organized into packages sharing all parameters except a_p
                                            if run == "linear":
                                                # runs all packages one after the other
                                                run_a_range(path, parameters, mode, e_p, mass_p, mu_p, phi_p, per, inc,
                                                            Omega, e_bin, mu_bin, phi_bin, n_orbits, dt, export)

                                                n = n + 1
                                                totaltime = timer() - start0
                                                averagetime = totaltime / n

                                                out = open(outputfile, "a")
                                                # calculates ETC and writes it as output, so you know, how long this will take
                                                out.write("___________________________________\n")
                                                out.write("current time: " + str(datetime.datetime.now()) + "\n")
                                                out.write("calculation {} of {}:\n".format(n, N))
                                                out.write("calc-time: {} min {:.2f} sec\n"
                                                          .format(np.floor((timer() - start) / 60),
                                                                  (timer() - start) % 60))
                                                out.write("total time: {} h {:.2f} min\n"
                                                          .format(np.floor(totaltime / 3600),
                                                                  (totaltime % 3600) / 60))
                                                out.write("ETA: {} h {:.2f} min\n"
                                                          .format(np.floor((averagetime * (N - n)) / 3600),
                                                                  ((averagetime * (N - n)) % 3600) / 60))
                                                out.write("___________________________________\n\n")
                                                out.close()

                                                start = timer()
                                            elif run == "parallel":
                                                # saves the arguments for simulation packages as list of tupels
                                                # to be able to run them in parallel later
                                                tupels.append((path, parameters, mode, e_p, mass_p, mu_p, phi_p, per,
                                                               inc, Omega, e_bin, mu_bin, phi_bin, n_orbits, dt, export))

                                            path = path_e_p
                                        path = path_mass_p
                                    path = path_mu_p
                                path = path_phi_p
                            path = path_per
                        path = path_Omega
                    path = path_inc
                path = path_e_bin
            path = path_mu_bin
        path = path_phi_bin

    # creates a multiprocessing pool to run all the simulation packages determined in the for-loop-monster
    if run == "parallel":
        with Pool(processes=cpus) as pool:
            pool.map(wrapper, tupels)


def wrapper(tupel):
    """Wrapper for the multiprocessing runs
    
    tupel -- Contains the arguments for simulation package, which are given to run_a_range
    """
    run_a_range(*tupel)

    # it's not the prettiest code, but it works
    global counter
    global start0
    global output

    # calculate and write ETC again
    n = counter.increment()
    totaltime = timer() - start0
    averagetime = totaltime / n
    try:
        out = open(output, "a")
    except:
        return

    out.write("___________________________________\n")
    out.write("current time: " + str(datetime.datetime.now()) + "\n")
    out.write("calculation {} done:\n".format(n))
    out.write("total time: {} h {:.2f} min\n".format(np.floor(totaltime / 3600),
                                                     (totaltime % 3600) / 60))
    out.write("___________________________________\n\n")
    out.close()


def run_a_range(path, parameters, mode, e_p, mass_p, mu_p, phi_p, per, inc, Omega, e_bin, mu_bin, phi_bin, n_orbits,
                dt, export):
    """Run a range of Rebound simulations sharing all parameters except a_p
    
    Arguments:
    path -- Path at which to store output
    parameters -- dictionary from run_parameter_range, included just in case we forgot something we need
    mode -- "1P" for single planet simulations; "2P" for planets in 1:1 resonance; "{x}:{y}" for planets in x:y resonance
    e_p -- planetary eccentricity (default 0.3)
    mass_p -- ratio of planet mass to binary mass ((m_1+m_2)/(M_1+M_2)) (default 1e-4)
    mu_p -- planetary mass ratio (m_2/(m_1+m_2)) (default 0.5)
    phi_p -- initial true anomaly of the second planet (default pi)
    per -- initial argument of pericenter (small omega) (default pi/2)
    inc -- planetary inclination (default 0)
    Omega -- planetary longitude of ascending node (default 0)
    e_bin -- binary eccentricity (default 0.04)
    mu_bin -- binary mass ratio (M_2/(M_1+M_2)) (default (0.4))
    phi_bin -- initial true anomaly of binary (default 0)
    n_orbits -- Number of binary orbits to be simulated (default 1e5)
    dt -- time step at which termination condition is checked and (if desired) output created (default 10)
    export -- determines output format ("txt", "sa", "pickle", ...) (default None)

    Almost all arguments are passed on to run_sim just like that
    """
    # Uses the last varied parameter as filename
    f = open(path[0:len(path) - 1] + ".acrit", "w+")

    # Initialization of the output file with all the given parameters
    f.write("e_p={:.4f}, mass_p={:e}, mu_p={:.4f}, phi_p={:.4f}*pi, "
            "per={:.4f}*pi, inc={:.4f}*pi, Omega = {:.4f}*pi, e_bin={:.4f}, "
            "mu_bin={:.4f}, phi_bin={:.4f}\n\n "
            .format(e_p, mass_p, mu_p, phi_p / np.pi, per / np.pi,
                    inc / np.pi, Omega / np.pi, e_bin, mu_bin, phi_bin / np.pi))
    f.write("a_p;t_surv;result\n")

    # run_sim wants semi-major axis a_p, but sometimes we specify distance of pericenter q_p instead (for eccentric studies)
    if parameters["a/q"] == "q":
        a_range = parameters["aq_p"] / (1 - e_p)
        # convert given values from q to a
    else:
        a_range = parameters["aq_p"]

    for i in range(len(parameters["aq_p"])):
        # loops through the values for a_p and runs the appropriate simulation
        a_p = a_range[i]

        filename = "a_p={:.5f}".format(a_p)

        surv, res = run_sim(mode, path, filename, a_p, e_p, mass_p, mu_p, phi_p,
                            per, inc, Omega, e_bin, mu_bin, phi_bin, n_orbits,
                            dt, export)

        # write survived_orbits into text file for determining stability maps later
        f.write("{:.4f};{:.1f};{}\n".format(a_p, surv, res))

    f.close()


def setup():
    """Setup script to ease creation of simulation config files
    
    Some values that need to be specified:
    mode -- "1P" for single planet simulations; "2P" for planets in 1:1 resonance; "{x}:{y}" for planets in x:y resonance
    e_p -- planetary eccentricity (default 0.3)
    mass_p -- ratio of planet mass to binary mass ((m_1+m_2)/(M_1+M_2)) (default 1e-4)
    mu_p -- planetary mass ratio (m_2/(m_1+m_2)) (default 0.5)
    phi_p -- initial true anomaly of the second planet [given in multiples of pi] (default 1)
    per -- initial argument of pericenter (small omega) [given in multiples of pi] (default 0.5)
    inc -- planetary inclination [given in multiples of pi] (default 0)
    Omega -- planetary longitude of ascending node [given in multiples of pi] (default 0)
    e_bin -- binary eccentricity (default 0.04)
    mu_bin -- binary mass ratio (M_2/(M_1+M_2)) (default (0.4))
    phi_bin -- initial true anomaly of binary [given in multiples of pi] (default 0)
    n_orbits -- Number of binary orbits to be simulated (default 1e5)
    dt -- time step at which termination condition is checked and (if desired) output created (default 10)
    export -- determines output format ("txt", "sa", "pickle", ...) (default None)

    Explanation of other abbreviations:
    var -- Introduce a variation of a parameter (will ask for lower and upper bounds as well as step size)
    val -- Change the value of a parameter
    vals -- Enter multiple values for a parameter as a list (if you don't/can't use equal spacing with var)
    """
    # Default values
    params = {
        "mu_bin": [0.4],
        "e_bin": [0.04],
        "mu_p": [0.5],
        "mass_p": [1e-4],
        "e_p": [0.3],
        "phi_bin": [0],
        "inc": [0],
        "Omega": [0],
        "per": [0.5],
        "phi_p": [1],
    }
    # List of Parameters which are given in multiples of pi
    piparams = ["phi_bin", "inc", "Omega", "per", "phi_p"]
    dt = 10

    mode = input("What mode shall be used? [2P, 1P]: ")
    if mode not in ["2P", "1P"] and ":" not in mode:
        print("Using default value [2P]")
        mode = "2P"

    aq = input("Semimajor axis or pericenter distance? [a, q]: ")
    if aq not in ["a", "q"]:
        print("Using default value [a]")
        aq = "a"

    valid = False
    while not valid:
        try:
            aq_p = [float(input("Enter lower bound for " + aq + "_p: "))]
            valid = True
        except ValueError:
            print("Invalid input")


    valid = False
    while not valid:
        try:
            aq_p.append(float(input("Enter upper bound for " + aq + "_p: ")))
            if aq_p[1] < aq_p[0]:
                print("Value can't be lower than " + str(aq_p[0]))
                continue
            valid = True
        except ValueError:
            print("Invalid input")


    valid = False
    while not valid:
        try:
            aq_p.append(float(input("Enter step size for " + aq + "_p: ")))
            if aq_p[2] <= 0:
                print("Value must be larger than 0")
                continue
            valid = True
        except ValueError:
            print("Invalid input")

    aq_p[2] = round((aq_p[1] - aq_p[0])/aq_p[2])+1

    finished = False
    while not finished:
        print("_________________________________________")
        print("Current values: ")
        print("Ranges given as [lower, upper, n_steps]")
        for param in params:
            if param in piparams:
                print(param + ": np.pi*" + str(params[param]))
            else:
                print(param + ": " + str(params[param]))
        print("aq_p: " + str(aq_p))
        print("a/q: " + aq)
        print("mode: " + mode)

        print("_________________________________________")

        print('To add a variation, type "var"')
        print('To change a value, type "val"')
        print('To set multiple values, type "vals"')
        print('To finish the setup, type "done"')
        command = input()

        if command == "done":
            finished = True
            break

        variable = input("Select parameter: ")
        if variable not in params:
            print("Selection invalid")
            continue

        if command == "val":
            try:
                params[variable] = [float(input("Enter value for " + variable + ": "))]
            except ValueError:
                print("Invalid input")
        elif command == "vals":
            params[variable] = input("Enter values as list: ")
        elif command == "var":
            tmp = params[variable]
            params[variable] = [-1, -1, -1]
            try:
                params[variable][0] = float(input("Enter lower bound for " + variable + ": "))
            except ValueError:  # in case user is dumb
                print("Invalid input")
                params[variable] = tmp
                continue

            try:
                params[variable][1] = float(input("Enter upper bound for " + variable + ": "))
                if params[variable][1] < params[variable][0]:
                    print("Value can't be lower than " + str(aq_p[0]))
                    params[variable] = tmp
                    continue
            except ValueError:
                print("Invalid input")
                params[variable] = tmp
                continue

            try:
                params[variable][2] = float(input("Enter step size for " + variable + ": "))
                if params[variable][2] <= 0:
                    print("Value must be larger than 0")
            except ValueError:
                print("Invalid input")
                params[variable] = tmp
                continue

            params[variable][2] = round((params[variable][1] - params[variable][0]) / params[variable][2]) + 1

    calcname = input("Enter calculation name: ") or "Botor_sim"

    valid = False
    while not valid:
        run = input("Parallelize run? [yes, no]: ") or "yes"
        if run == "yes":
            run = "parallel"
            valid = True
            break
        elif run == "no":
            run = "linear"
            valid = True
            break
        else:
            print("Unrecognized input, try again.")

    # Before we implemented multiprocessing, we used to split the configuration in multiple files
    # which we started separately. This is shouldn't be necessary to use anymore
    split = False
    splitvar = "e_p"
    nsplit = 1
    if run == "linear":
        valid = False
        while not valid:
            split = input("Split calculation? [yes, no]: ") or "no"
            if split == "yes":
                split = True
            else:
                split = False
                valid = True
                break

            if split:
                splitvar = input("Enter variable to split: ") or "e_p"
                if splitvar not in params:
                    print("Invalid selection")
                    continue
                elif not len(params[splitvar]) == 3:
                    print("Enter parameter with a specified range!")
                    continue

                try:
                    nsplit = int(input("Enter number of splits: ") or 8)
                except ValueError:
                    print("Invalid input, using 5")
                    nsplit = 5
                    valid = True
                    break
                if nsplit <= 0:
                    print("Invalid input, using 5")
                    nsplit = 5
                    valid = True
                    break
                else:
                    valid = True
                    break

    # Choice of output for simulation data:
    # txt -- Text file
    # sa -- Rebound SimulationArchive
    # pickle -- Python Pickle file
    valid = False
    out = "None"
    while not valid:
        out = input("Output format (None):") or "None"
        if out in [None, "none", "None"]:
            out = "None"
            valid = True
            break
        elif out not in ["txt", "sa", "pickle"]:
            print("Invalid input")
            continue
        else:
            out = '"' + out + '"'
            try:
                dt = float(input("Output data every X binary orbits [10]: ") or 10)
                if dt <= 0:
                    print("Invalid input")
                else:
                    valid = True
                    break
            except ValueError:
                print("Invalid input")

    valid = False
    while not valid:
        path = input("Enter path [/scratch/adelbert/]: ") or "/scratch/adelbert/"
        if not os.path.exists(path):
            print("Path doesn't exist")
            continue
        valid = True

    path = path + calcname + "/"
    os.makedirs(path + "output", exist_ok=True)
    os.makedirs(path + "setup", exist_ok=True)
    os.makedirs(path + "data", exist_ok=True)

    """
    What follows creates a submit script for different queueing programs we used (slurm, tsp, ...)
    But these depend on your machine, so you'll have to figure this out yourself, sry.
    
    f = open(path + "submit.job", "w+")
    # f.write("#!/bin/bash\n")
    # f.write("#SBATCH --partition=cpu\n")
    # f.write("#SBATCH -J " + calcname + "\n")
    # f.write("#SBATCH --time=07-00\n")
    # f.write("#SBATCH --mail-type=ALL\n")
    # f.write("#SBATCH --mail-user=stefan.adelbert@uni-tuebingen.de\n")
    # f.write("#SBATCH --output=" + path + "output/slurm-%j.out\n")
    # f.write("#SBATCH --array=0-" + str(nsplit-1) + "\n\n")

    # f.write("source ~/adelbert/Code/venv/bin/activate\n\n")

    # f.write("FILES=(" + path + "setup/*)\n")
    # f.write("python3 ${FILES[$SLURM_ARRAY_TASK_ID]}\n\n")

    # f.write("deactivate")

    f.write("#!/bin/bash\n\n")
    f.write("for file in ./setup/*.py\n")
    f.write("do\n")
    f.write('  tsp -L ' + calcname + ' python3 "$file"\n')
    f.write("done")

    f.close()

    os.system("chmod +x " + path + "submit.job")"""

    if split:
        split_arrays = np.array_split(np.linspace(params[splitvar][0], params[splitvar][1],
                                                  np.round(params[splitvar][2])), nsplit)

    for k in range(nsplit):
        # Since the splitting method is outdated, this loop should only run once.
        # It creates a .py file defining the parameters dictionary used above
        # and containing the function call to run_parameter_range.
        # Simply running that file with "python3 run_calc1.py" should start the calculation
        # however you may need to fiddle around with the import, so it can actually access botor.py
        f = open(path + "setup/run_calc" + str(k) + ".py", "w+")
        f.write("import numpy as np\n")
        f.write("from botor import run_parameter_range\n\n")

        f.write('parameters = {\n')
        for param in params:
            if split and param == splitvar:
                continue
            value = params[param]
            if len(value) == 1 and param in piparams:
                f.write('\t"'+param+'": [np.pi*' + str(value[0]) + '],\n')
            elif len(value) == 3 and param in piparams:
                f.write('\t"'+param+'": np.pi*np.linspace(' + str(value[0]) + ',' + str(value[1]) + ','
                        + str(value[2]) + '),\n')
            elif param in piparams:
                f.write('\t"' + param + '": np.pi*np.linspace(' + value + '),\n')
            elif len(value) == 1:
                f.write('\t"'+param+'": ' + str(value) + ',\n')
            elif len(value) == 3:
                f.write('\t"'+param+'": np.linspace(' + str(value[0]) + ',' + str(value[1]) + ','
                        + str(value[2]) + '),\n')
            else:
                f.write('\t"' + param + '": ' + value + ',\n')

        if split:
            if splitvar in piparams:
                f.write('\t"'+splitvar+'": np.pi*np.array('+np.array2string(split_arrays[k], separator=',')+'),\n')
            else:
                f.write('\t"'+splitvar+'": '+np.array2string(split_arrays[k], separator=',')+',\n')

        f.write('\t"aq_p": np.linspace(' + str(aq_p[0]) + ',' + str(aq_p[1]) + ',' + str(aq_p[2]) + '),\n')

        f.write('\t"a/q": "' + aq + '",\n')
        f.write('\t"mode": "' + mode + '"\n}\n\n')

        f.write('run_parameter_range(parameters, "' + path + 'data/", "'
                + path + 'output/' + calcname + "_" + str(k) + '.out", export = ' + out + ', dt = ' + str(dt)
                + ', run = "' + run + '")')
        f.close()

    print("Setup complete")








