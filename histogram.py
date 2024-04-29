import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
import warnings


def histogram(model, step, lifted=""):  # model is passed in with .npy
    plt.clf()
    if lifted == "":
        path = f"n_step_norms/{model}/data/s{step}_norms_all.npy"
    else:
        path = f"n_step_norms/{model}/data/s{step}_norms_{lifted}_all.npy"

    # file_path = os.path.join(path, ".npy")
    norms_all = np.load(path)
    print(norms_all.shape)

    norms_all = norms_all.flatten()
    norms_all_weights = np.ones_like(norms_all) / len(norms_all)

    plot_path = f"n_step_norms/{model}/plots/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Create the histogram
    plt.hist(
        norms_all, weights=norms_all_weights
    )  # You can adjust the number of bins as needed
    if lifted == "":
        plt.title(f"Histogram of {step}-Step Norms")
        plot_name = f"s{step}_norms_all"
    else:
        plt.title(f"Histogram of {step}-Step Norms ({lifted})")
        plot_name = f"s{step}_norms_{lifted}_all"
    plt.xlabel("Norms")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(plot_path, plot_name))


def step_predictions(cfg_name, obs="POLY", obs_p=10):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read("config.cfg")
    args = dict(config[cfg_name])
    data_path = f"data/XAllReal_nb_{args['n_buffer']}_cb_{args['c_buffer']}.npy"

    model_name = f"nb_{args['n_buffer']}_cb_{args['c_buffer']}_obs_{obs}_{obs_p}_reg_EDMDc"  # have to change this value

    model_path = os.path.join("models", f"{model_name}.npy")
    model_npy = np.load(model_path, allow_pickle=True)
    model = model_npy.item()
    # print(type(model))

    Xfull = np.load(data_path, allow_pickle=True)
    print(Xfull.shape)  # (5, 101, 2000)

    N_T = Xfull.shape[1]
    N_x = int(args["N_x"])
    N_u = int(args["N_u"]) * int(args["c_buffer"])
    print(N_T, N_x, N_u)

    s1_norms_all = []
    s5_norms_all = []
    s10_norms_all = []

    for i in range(Xfull.shape[2]):  # data.shape[2]
        s1_norms = []
        s5_norms = []
        s10_norms = []
        for j in range(Xfull.shape[1] - 1):  # data.shape[1] - 1
            x1 = Xfull[:N_x, j, i]
            u = Xfull[N_x : N_x + N_u, j, i]
            x2_true = Xfull[-N_x:, j + 1, i]

            # 1 step
            x2_sim = model.predict(x1.reshape(1, -1), u.reshape(1, -1))
            # print(x2_sim.shape, x2_true.reshape(1, -1).shape)
            x2_sim_ = model.observables.transform(x2_sim)
            x2_true_ = model.observables.transform(x2_true.reshape(1, -1))

            # print(x2_sim_.shape, x2_true_.shape)
            x2_err = np.linalg.norm(x2_sim_ - x2_true_)  # 2-norm of x2_sim and x2_true
            # print(x2_err)
            s1_norms.append(x2_err)

            for ts in range(1, 10, 1):
                if j + ts < N_T:
                    u_ts = Xfull[N_x : N_x + N_u, j + ts, i]
                    # transform, get A,
                    # lift and then roll out (already being done EXCEPT need to keep it in lifted state)
                    # lift later data
                    x2_sim = model.predict(x2_sim, u_ts.reshape(1, -1))
                    x2_sim_ = model.observables.transform(x2_sim)
                    if ts == 4 and (j + 5 < N_T):  # time step 5 has been simulated
                        x2_true = Xfull[:N_x, j + 5, i]
                        x2_true_ = model.observables.transform(x2_true.reshape(1, -1))
                        x2_err = np.linalg.norm(
                            x2_sim_ - x2_true_
                        )  # 2-norm of x2_sim and x2_true
                        s5_norms.append(x2_err)
                    if ts == 9 and (j + 10 < N_T):  # time step 10 has been simulated
                        x2_true = Xfull[:N_x, j + 10, i]
                        x2_true_ = model.observables.transform(x2_true.reshape(1, -1))
                        x2_err = np.linalg.norm(
                            x2_sim_ - x2_true_
                        )  # 2-norm of x2_sim and x2_true
                        s10_norms.append(x2_err)
        s1_norms_all.append(s1_norms)
        s5_norms_all.append(s5_norms)
        s10_norms_all.append(s10_norms)

    s1_norms_all = np.array(s1_norms_all)
    s5_norms_all = np.array(s5_norms_all)
    s10_norms_all = np.array(s10_norms_all)

    base_path = f"n_step_norms/{model_name}/data"

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    np.save(os.path.join(base_path, "s1_norms_lifted_all.npy"), s1_norms_all)
    np.save(os.path.join(base_path, "s5_norms_lifted_all.npy"), s5_norms_all)
    np.save(os.path.join(base_path, "s10_norms_lifted_all.npy"), s10_norms_all)

    histogram(model_name, 1, "lifted")
    histogram(model_name, 5, "lifted")
    histogram(model_name, 10, "lifted")


# step_predictions("REAL_NB_15_CB_5")

# histogram("VanderPol_rbf", 1, "lifted")
# histogram("VanderPol_rbf", 5, "lifted")
# histogram("VanderPol_rbf", 10, "lifted")
