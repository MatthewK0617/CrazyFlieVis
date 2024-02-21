# Import necessary libraries
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

    # Remove all unnecessary values
    initial_state = df.iloc[0].val
    start_index = 0
    for i in range(0, len(df)):
        curr_row = df.iloc[i]
        if curr_row.type == "state":
            if np.any(np.not_equal(initial_state, curr_row.val)):
                start_index = i - 2
                break
            else:
                continue
    df = df.iloc[start_index:]

    # Append df data to X
    for i in range(1, len(df) - (2*b-1)):
        if df.iloc[i].type == "control":
            # TODO add check for ensuring stacking errors do not occur
            if df.iloc[i - 1].type == "state" and df.iloc[i + (2*b-1)].type == "state":
                x_pre = df.iloc[i - 1].val
                c = df.iloc[i].val
                x_pos = df.iloc[i + b].val
                temp = np.concatenate((x_pre, c, x_pos))
                # print(temp)
                X.append(temp)
            else:
                continue

    X = np.array(X)

    Xfull = [X]  # fix this
    Xfull = np.array(Xfull)
    Xfull = Xfull.transpose(2, 1, 0)
    Xfull = Xfull[:, 500:1500, :]

    print(Xfull.shape)
    np.save(f"data/Xfull_{b}_500-1500", Xfull)

    return Xfull


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
    plt.savefig("plots/plot_df.png")
    # plt.show()


def plot_np_data(X, b=1):
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
    plt.savefig(f"plots/plot_np_{b}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot states and controls from a ROS bag."
    )
    parser.add_argument(
        "-bp",
        default="raw_data/test3_2024-02-08-17-44-08.bag",
        type=str,
        help="Path to the ROS bag file",
    )
    args = parser.parse_args()

    df = ros_messages_to_dataframe(args.bp)
    plot_df_data(df)

    buffer = 10
    Xfull = df_to_np(df, buffer)
    plot_np_data(Xfull, buffer)
