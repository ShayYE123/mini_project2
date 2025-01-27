import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points_file, ecog_data_file):

    # reading the "events_file_ordered.csv" while making sure that the data collected from each column is integer.
    trial_points = pd.read_csv(trial_points_file, names=["start", "peak", "finger"], dtype={"start": int, "peak": int, "finger": int})

    # reading the "brain_data_channel_one.csv", and turning the data in the column into a numpy array that will be easier to work with.
    ecog_data = pd.read_csv(ecog_data_file).squeeze("columns").values

    # Define constants of matrix, so that it will collect the 200 events before begining of finger movement, and 1000 events after the begining of finger movement.
    pre_event = 200
    post_event = 1000
    total_points = pre_event + 1 + post_event

    # Initialize a matrix to store the average ERP for each finger (5 rows, 1201 columns)
    fingers_erp_mean = np.zeros((5, total_points))

    # Process each finger (1 to 5)
    for finger in range(1, 6):
        # Filter trials for the current finger
        finger_trials = trial_points[trial_points["finger"] == finger]

        # Collect ERP signals for all trials of the current finger
        erp_signals = []
        for _, row in finger_trials.iterrows():
            start_idx = row["start"]
            # Extract the block of data for this trial
            block = ecog_data[start_idx - pre_event : start_idx + post_event + 1]
            if len(block) == total_points:  # Ensure block has the correct length
                erp_signals.append(block)

        # Compute the mean ERP for the current finger
        if erp_signals:
            fingers_erp_mean[finger - 1, :] = np.mean(erp_signals, axis=0)

    # Plot the averaged brain responses for all five fingers
    plt.figure(figsize=(12, 8))
    for finger in range(1, 6):
        plt.plot(fingers_erp_mean[finger - 1, :], label=f"Finger {finger}")
    plt.title("Averaged Brain Responses for Each Finger Movement")
    plt.xlabel("Time (ms)")
    plt.ylabel("Brain Signal Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the matrix
    print("Fingers ERP Mean Matrix:")
    for i, row in enumerate(fingers_erp_mean, start=1):
        print(f"Finger {i}: {row}")

    return fingers_erp_mean


# Example usage
trial_points = "events_file_ordered.csv"
ecog_data = "brain_data_channel_one.csv"
fingers_erp_mean = calc_mean_erp(trial_points, ecog_data)
