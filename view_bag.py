# Import necessary libraries
from os import listdir
from os.path import isfile, join
import rosbag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import argparse  # Import argparse for command-line parsing


# Function to convert ROS messages to pandas DataFrame
def ros_messages_to_dataframe(bag_path, topics=["/state", "/control/final"]):
    # Initialize an empty list to store the data
    data = []

    # Open the ROS bag file
    bag = rosbag.Bag(bag_path)

    # Iterate over each message in the specified topic
    for msg in bag.read_messages(topics=topics):
        # Convert the ROS time to a datetime object
        timestamp = datetime.fromtimestamp(msg.timestamp.to_sec())
        # Extract the data fields you're interested in (e.g., 'state' and 'control')
        # This assumes the message has attributes 'state' and 'control'
        if msg.topic == "/state":
            data.append(
                {
                    "timestamp": timestamp,
                    "type": "state",
                    "val": np.array(msg.message.state.__getstate__()),
                }
            )
        elif msg.topic == "/control/final":
            data.append(
                {
                    "timestamp": timestamp,
                    "type": "control",
                    "val": np.array(msg.message.control.__getstate__()),
                }
            )
        # data.append({'timestamp': timestamp, 'state': msg.state, 'control': msg.control})

    # Close the bag file
    bag.close()

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(data)

    # Set the timestamp as the index
    df.set_index("timestamp", inplace=True)
    # print(df.iloc[404], df.iloc[406])

    # Return the file
    return df


def df_to_np(df, b=1):
    X = []  # states, time points, num trajectories
    # Remove all stationary values
    e = 0.05
    initial_state = df.iloc[0]
    init_x = initial_state.val[0]
    init_y = initial_state.val[1]
    start_index = 0
    initial_length = len(df)

    # Remove take off data
    for i in range(0, len(df)):
        curr_row = df.iloc[i]
        x = curr_row.val[0]
        y = curr_row.val[1]
        # print(curr_row.val)
        if curr_row.type == "state":
            if abs(init_y - y) > e or abs(init_x - x) > e:
                start_index = max(i - 20, start_index)
                break
    df = df.iloc[start_index:]
    second_length = len(df)

    # Remove landing data
    end_index = len(df) - 1
    xf = df.iloc[end_index].val[0]
    yf = df.iloc[end_index].val[1]
    for i in range(len(df) - 1, -1, -1):
        curr_row = df.iloc[i]
        x = curr_row.val[0]
        y = curr_row.val[1]
        if curr_row.type == "state":
            if abs(xf - x) > e or abs(yf - y) > e:
                end_index = min(i + 20, end_index)
                print(end_index)
                break
    df = df.iloc[:end_index]
    # plot_df_data(df)
    print(initial_length, second_length, len(df))

    # Append df data to X
    i = 1
    while i < len(df) - (2 * b - 1):
        if df.iloc[i].type == "control":
            if (
                df.iloc[i - 1].type == "state"
                and df.iloc[i + (2 * b - 1)].type == "state"
            ):
                x_pre = df.iloc[i - 1].val
                c = df.iloc[i].val
                x_pos = df.iloc[i + (2 * b - 1)].val
                temp = np.concatenate((x_pre, c, x_pos))
                X.append(temp)
                i += 2 * b - 1  # Skip ahead
            else:
                i += 1  # Move to the next iteration
        else:
            i += 1  # Move to the next iteration

    X = np.array(X)

    Xfull = [X]  # fix this
    Xfull = np.array(Xfull)
    Xfull = Xfull.transpose(2, 1, 0)

    return Xfull


def np_stacked_controls(X, b=1, cb=1, N_x=7, N_u=4):
    """
    cb for control buffer
    """
    if b < cb:
        print(
            "Buffer multiplier must be greater than or equal to the control stack multiplier."
        )
        return
    # stack controls
    X_stacked = []
    for i in range(0, X.shape[1] - b, b):
        x_stacked = []
        x_stacked.extend(X[:N_x, i])

        count = 0
        while count < cb:
            x_stacked.extend(X[N_x : N_x + N_u, i + count])
            count += 1
        x_stacked.extend(X[N_x + N_u :, i + b])
        x_stacked = np.array(x_stacked)
        X_stacked.append(x_stacked)

    X_stacked = np.array(X_stacked)
    X_stacked = X_stacked.transpose(1, 0, 2)

    return X_stacked


def interpolate_anomalies(arr, threshold=0.01):
    """
    Interpolates anomalies in a numpy array based on a defined threshold.
    arr: Input array of shape (features, time_points).
    threshold: A threshold for detecting anomalies based on sudden changes.
    """
    # Assuming arr is 2D: features x time_points
    for i in range(1, arr.shape[1] - 1):  # Skip the first and last points
        prev_point = arr[:, i - 1]
        curr_point = arr[:, i]
        next_point = arr[:, i + 1]

        # Calculate the change magnitude between points
        change_magnitude = np.abs(curr_point - prev_point)
        next_change_magnitude = np.abs(next_point - curr_point)

        # Detect anomalies based on the threshold
        if np.any(change_magnitude > threshold) or np.any(
            next_change_magnitude > threshold
        ):
            # Interpolate the current point
            arr[:, i] = (prev_point + next_point) / 2

    return arr


def combine_trajectories(nb=1, cb=1):
    dir = "real_raw_data/"
    filenames = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    n_trajectories = len(filenames)
    all_trajectories = []
    for i in range(n_trajectories):
        print(filenames[i])
        df = ros_messages_to_dataframe(filenames[i])
        arr = df_to_np(df, 1)
        print(arr.shape)
        arr = np_stacked_controls(arr, b=nb, cb=cb)[:, :, 0]
        # should check for anomalies and iterpolate here
        # print(arr.shape)
        arr = interpolate_anomalies(arr, threshold=0.1)
        all_trajectories.append(arr)

    min_length = min(len(traj[1]) for traj in all_trajectories)
    all_trajectories = [traj[:, :min_length] for traj in all_trajectories]

    # Convert list of arrays into a single NumPy array
    all_trajectories = np.array(all_trajectories)
    all_trajectories = all_trajectories.transpose(1, 2, 0)

    return all_trajectories


# Plotting function
def plot_df_data(df):
    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    df_state = df.loc[df["type"] == "state"]
    colors = mpl.cm.viridis(np.linspace(0, 1, len(df_state)))
    ax.scatter(
        [r[0] for r in df_state.val],
        [r[1] for r in df_state.val],
        [r[2] for r in df_state.val],
        color=colors,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2)
    df_control = df.loc[df["type"] == "control"]
    for i, name in enumerate(["roll", "pitch", "yaw_dot", "thrust"]):
        ax2.plot(df_control.index, [r[i] for r in df_control.val], label=name)
    ax2.set_xlabel("Time")

    plt.title("State and Control over Time")
    plt.legend()
    # plt.savefig("plots/plot_df_delete.png")
    plt.show()


def plot_np_data(X, b=1, cb=1):
    fig = plt.figure(figsize=(10, 6))

    # Assuming each of x_pre, c, and x_pos have 3 elements
    state_length = 7  # Length of the state vector (x_pre and x_pos)
    control_length = 4  # Length of the control vector (c)

    # Extracting states and controls
    states = X[:state_length, :, 0].T
    controls = X[state_length : state_length + control_length, :, 0].T
    print(states.shape)

    # Plotting states
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    colors = mpl.cm.viridis(np.linspace(0, 1, len(states)))
    ax.scatter(states[:, 0], states[:, 1], states[:, 2], color=colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Plotting controls
    ax2 = fig.add_subplot(1, 2, 2)
    for i, name in enumerate(
        ["roll", "pitch", "yaw_dot", "thrust"]
    ):  # Adjust control names as needed
        ax2.plot(np.arange(len(controls)), controls[:, i], label=name)
    ax2.set_xlabel("Time")

    plt.title("State and Control over Time")
    plt.legend()
    # plt.savefig(f"plots/plot_np_nbuffer_{b}_cbuffer_{cb}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot states and controls from a ROS bag."
    )
    parser.add_argument(
        "-bp",
        default="real_raw_data/2024-02-26-19-37-50.bag",
        type=str,
        help="Path to the ROS bag file",
    )
    parser.add_argument(
        "-nb",
        default=1,
        type=int,
        help="State buffer",
    )
    parser.add_argument(
        "-cb",
        default=1,
        type=int,
        help="Control stack parameter",
    )

    args = parser.parse_args()
    # df = ros_messages_to_dataframe(args.bp)
    # plot_df_data(df)

    nb = args.nb
    cb = args.cb
    all_trajectories = combine_trajectories(nb, cb)
    np.save(f"data/XAllReal_nb_{nb}_cb_{cb}", all_trajectories)
    print(all_trajectories.shape)
