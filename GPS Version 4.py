import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load the JSON data
with open('All_players.json') as file:
    data = json.load(file)

# Extract the raw data for each athlete from the JSON
raw_data_dict = {player['name'].strip(): player['raw'] for player in data['c']}

# Create a dictionary to store the height and weight information for each athlete
athlete_info = {
    'Peitsis V': {'height': 1.80, 'weight': 75.0},
    'Leka M': {'height': 1.75, 'weight': 70.0},
    'Memetoglou O': {'height': 1.85, 'weight': 80.0},
    'Karypidis S': {'height': 1.78, 'weight': 72.0}
}

# Define the sprint start threshold
sprint_threshold = 0.25  # Sprint start threshold velocity difference in m/s

# Create a list to strore data for all players
all_players_data=[]

# Process each athlete's data
for athlete_name, raw_data in raw_data_dict.items():
    sprint_data = []
    athlete_height = athlete_info[athlete_name.strip()]['height']
    athlete_weight = athlete_info[athlete_name.strip()]['weight']

    # Preprocess the data
    distance = 0
    for i in range(len(raw_data) - 1):
        current_data = raw_data[i]
        next_data = raw_data[i + 1]

        # Convert timestamp from UNIX to datetime object
        timestamp = datetime.datetime.fromtimestamp(current_data['stamp'])

        # Get velocity and acceleration from the data
        velocity = current_data['speed']
        acceleration = current_data['gps_acc']

        # Calculate distance
        time_diff = next_data['stamp'] - current_data['stamp']
        distance += velocity * time_diff

        sprint_data.append({
            'timestamp': timestamp,
            'velocity': velocity,
            'acceleration': acceleration,
            'distance': distance
        })

    # Determine the start of the sprint based on the velocity difference threshold
    sprint_start_index = next((i for i in range(len(sprint_data) - 1) if sprint_data[i + 1]['velocity'] - sprint_data[i]['velocity'] >= sprint_threshold), None)

    if sprint_start_index is None:
        print(f"No sprint detected for {athlete_name} based on the given threshold.")
        continue

    sprint_data = sprint_data[sprint_start_index:]

    # Find the maximum velocity and its index
    max_velocity = max(data['velocity'] for data in sprint_data)
    max_velocity_index = next(i for i, data in enumerate(sprint_data) if data['velocity'] == max_velocity)

    # Find the index of the data point where velocity starts decreasing after the maximum velocity
    decreasing_velocity_index = next(i for i in range(max_velocity_index, len(sprint_data) - 1) if sprint_data[i + 1]['velocity'] < sprint_data[i]['velocity'])

    # Find the minimum velocity during the late deceleration phase
    late_deceleration_data = sprint_data[decreasing_velocity_index:]
    min_velocity = min(data['velocity'] for data in late_deceleration_data)
    min_velocity_index = next(i for i, data in enumerate(late_deceleration_data) if data['velocity'] == min_velocity)

    # Update the sprint data to end at the minimum velocity during late deceleration
    sprint_data = sprint_data[:decreasing_velocity_index + min_velocity_index + 1]

    # Calculate kinematic variables
    deceleration_data = sprint_data[max_velocity_index:]
    deceleration_time = (deceleration_data[-1]['timestamp'] - deceleration_data[0]['timestamp']).total_seconds()
    deceleration_distance = sum(data['velocity'] * (next_data['timestamp'] - data['timestamp']).total_seconds() for data, next_data in zip(deceleration_data[:-1], deceleration_data[1:]))

    avg_deceleration = (max_velocity - min_velocity) / deceleration_time
    max_deceleration = min(data['acceleration'] for data in deceleration_data)

    mid_velocity = 0.5 * max_velocity
    mid_velocity_index = next(i for i, data in enumerate(deceleration_data) if data['velocity'] <= mid_velocity)

    early_deceleration_data = deceleration_data[:mid_velocity_index]
    late_deceleration_data = deceleration_data[mid_velocity_index:]

    early_avg_deceleration = (max_velocity - mid_velocity) / (early_deceleration_data[-1]['timestamp'] - early_deceleration_data[0]['timestamp']).total_seconds()
    late_avg_deceleration = (mid_velocity - min_velocity) / (late_deceleration_data[-1]['timestamp'] - late_deceleration_data[0]['timestamp']).total_seconds()

    time_to_mid_velocity = (early_deceleration_data[-1]['timestamp'] - early_deceleration_data[0]['timestamp']).total_seconds()
    time_to_max_deceleration = (deceleration_data[np.argmin([data['acceleration'] for data in deceleration_data])]['timestamp'] - deceleration_data[0]['timestamp']).total_seconds()

    # Calculate kinetic variables
    avg_horizontal_braking_force = -athlete_weight * avg_deceleration
    max_horizontal_braking_force = -athlete_weight * max_deceleration
    early_avg_horizontal_braking_force = -athlete_weight * early_avg_deceleration
    late_avg_horizontal_braking_force = -athlete_weight * late_avg_deceleration

    avg_horizontal_braking_power = avg_horizontal_braking_force * np.mean([data['velocity'] for data in deceleration_data])
    max_horizontal_braking_power = max_horizontal_braking_force * max_velocity
    early_avg_horizontal_braking_power = early_avg_horizontal_braking_force * np.mean([data['velocity'] for data in early_deceleration_data])
    late_avg_horizontal_braking_power = late_avg_horizontal_braking_force * np.mean([data['velocity'] for data in late_deceleration_data])

    avg_horizontal_braking_impulse = avg_horizontal_braking_force * deceleration_time
    max_horizontal_braking_impulse = max_horizontal_braking_force * deceleration_time
    early_avg_horizontal_braking_impulse = early_avg_horizontal_braking_force * (early_deceleration_data[-1]['timestamp'] - early_deceleration_data[0]['timestamp']).total_seconds()
    late_avg_horizontal_braking_impulse = late_avg_horizontal_braking_force * (late_deceleration_data[-1]['timestamp'] - late_deceleration_data[0]['timestamp']).total_seconds()

    # Calculate additional metrics
    segment_size = 5  # Segment size in meters
    segments = [sprint_data[0]['distance']]
    while segments[-1] + segment_size < sprint_data[-1]['distance']:
        segments.append(segments[-1] + segment_size)

    segment_metrics = []
    segment_times = []
    for i in range(len(segments) - 1):
        start_distance = segments[i]
        end_distance = segments[i + 1]
        segment_data = [data for data in sprint_data if start_distance <= data['distance'] < end_distance]

        if segment_data:
            mean_acceleration = np.mean([data['acceleration'] for data in segment_data])
            peak_acceleration = max(data['acceleration'] for data in segment_data)
            mean_velocity = np.mean([data['velocity'] for data in segment_data])
            peak_velocity = max(data['velocity'] for data in segment_data)
            start_time = segment_data[0]['timestamp']
            end_time = segment_data[-1]['timestamp']
            segment_time = (end_time - start_time).total_seconds()

            segment_metrics.append({
                'segment': f"{start_distance:.2f}m - {end_distance:.2f}m",
                'mean_acceleration': mean_acceleration,
                'peak_acceleration': peak_acceleration,
                'mean_velocity': mean_velocity,
                'peak_velocity': peak_velocity
            })

            segment_times.append({
                'segment': f"{start_distance:.2f}m - {end_distance:.2f}m",
                'time': segment_time
            })

    half_max_velocity_data = [data for data in sprint_data if data['velocity'] <= 0.5 * max_velocity]
    half_max_velocity_mean_acceleration = np.mean([data['acceleration'] for data in half_max_velocity_data])
    half_max_velocity_peak_acceleration = max(data['acceleration'] for data in half_max_velocity_data)

    # Export results to CSV file
    csv_filename = f"{athlete_name}_deceleration_results.csv"
    fieldnames = ['Variable', 'Value']

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Variable': 'Max Velocity (m/s)', 'Value': f'{max_velocity:.2f}'})
        writer.writerow({'Variable': 'Min Velocity (m/s)', 'Value': f'{min_velocity:.2f}'})
        writer.writerow({'Variable': 'Average Deceleration (m/s^2)', 'Value': f'{avg_deceleration:.2f}'})
        writer.writerow({'Variable': 'Early Average Deceleration (m/s^2)', 'Value': f'{early_avg_deceleration:.2f}'})
        writer.writerow({'Variable': 'Late Average Deceleration (m/s^2)', 'Value': f'{late_avg_deceleration:.2f}'})
        writer.writerow({'Variable': 'Maximum Deceleration (m/s^2)', 'Value': f'{max_deceleration:.2f}'})
        writer.writerow({'Variable': 'Deceleration Time (s)', 'Value': f'{deceleration_time:.2f}'})
        writer.writerow({'Variable': 'Time to Mid Velocity (s)', 'Value': f'{time_to_mid_velocity:.2f}'})
        writer.writerow({'Variable': 'Time to Max Deceleration (s)', 'Value': f'{time_to_max_deceleration:.2f}'})
        writer.writerow({'Variable': 'Deceleration Distance (m)', 'Value': f'{deceleration_distance:.2f}'})
        writer.writerow({'Variable': 'Average Horizontal Braking Force (N)', 'Value': f'{avg_horizontal_braking_force:.2f}'})
        writer.writerow({'Variable': 'Early Average Horizontal Braking Force (N)', 'Value': f'{early_avg_horizontal_braking_force:.2f}'})
        writer.writerow({'Variable': 'Late Average Horizontal Braking Force (N)', 'Value': f'{late_avg_horizontal_braking_force:.2f}'})
        writer.writerow({'Variable': 'Maximum Horizontal Braking Force (N)', 'Value': f'{max_horizontal_braking_force:.2f}'})
        writer.writerow({'Variable': 'Average Horizontal Braking Power (W)', 'Value': f'{avg_horizontal_braking_power:.2f}'})
        writer.writerow({'Variable': 'Early Average Horizontal Braking Power (W)', 'Value': f'{early_avg_horizontal_braking_power:.2f}'})
        writer.writerow({'Variable': 'Late Average Horizontal Braking Power (W)', 'Value': f'{late_avg_horizontal_braking_power:.2f}'})
        writer.writerow({'Variable': 'Maximum Horizontal Braking Power (W)', 'Value': f'{max_horizontal_braking_power:.2f}'})
        writer.writerow({'Variable': 'Average Horizontal Braking Impulse (N·s)', 'Value': f'{avg_horizontal_braking_impulse:.2f}'})
        writer.writerow({'Variable': 'Early Average Horizontal Braking Impulse (N·s)', 'Value': f'{early_avg_horizontal_braking_impulse:.2f}'})
        writer.writerow({'Variable': 'Late Average Horizontal Braking Impulse (N·s)', 'Value': f'{late_avg_horizontal_braking_impulse:.2f}'})
        writer.writerow({'Variable': 'Maximum Horizontal Braking Impulse (N·s)', 'Value': f'{max_horizontal_braking_impulse:.2f}'})
        writer.writerow({'Variable': 'Half Max Velocity Mean Acceleration (m/s^2)', 'Value': f'{half_max_velocity_mean_acceleration:.2f}'})
        writer.writerow({'Variable': 'Half Max Velocity Peak Acceleration (m/s^2)', 'Value': f'{half_max_velocity_peak_acceleration:.2f}'})

        for metric in segment_metrics:
            writer.writerow({'Variable': f"{metric['segment']} Mean Acceleration (m/s^2)", 'Value': f"{metric['mean_acceleration']:.2f}"})
            writer.writerow({'Variable': f"{metric['segment']} Peak Acceleration (m/s^2)", 'Value': f"{metric['peak_acceleration']:.2f}"})
            writer.writerow({'Variable': f"{metric['segment']} Mean Velocity (m/s)", 'Value': f"{metric['mean_velocity']:.2f}"})
            writer.writerow({'Variable': f"{metric['segment']} Peak Velocity (m/s)", 'Value': f"{metric['peak_velocity']:.2f}"})

        for segment in segment_times:
            writer.writerow({'Variable': f"{segment['segment']} Time (s)", 'Value': f"{segment['time']:.2f}"})

    print(f"Results exported to {csv_filename}")

    
    # Create graphs
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"{athlete_name} Deceleration Analysis")

    # Velocity-Time graph
    start_time = sprint_data[0]['timestamp']
    x_values = [(data['timestamp'] - start_time).total_seconds() for data in sprint_data]
    ax1.plot(x_values, [data['velocity'] for data in sprint_data], color='blue', label='Velocity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)', color='blue')
    ax1.tick_params('y', colors='blue')

    # Acceleration-Time graph
    ax2 = ax1.twinx()
    ax2.plot(x_values, [data['acceleration'] for data in sprint_data], color='red', label='Acceleration')
    ax2.set_ylabel('Acceleration (m/s^2)', color='red')
    ax2.tick_params('y', colors='red')


    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # Save the graph
    graph_filename = f"{athlete_name}_deceleration_analysis.png"
    plt.savefig(graph_filename)
    print(f"Graph saved as {graph_filename}")

    plt.close()

    # Append the athlete's data to the all_players_data list
    all_players_data.append({
        'Athlete': athlete_name,
        'Max Velocity (m/s)': max_velocity,
        'Min Velocity (m/s)': min_velocity,
        'Average Deceleration (m/s^2)': avg_deceleration,
        'Early Average Deceleration (m/s^2)': early_avg_deceleration,
        'Late Average Deceleration (m/s^2)': late_avg_deceleration,
        'Maximum Deceleration (m/s^2)': max_deceleration,
        'Deceleration Time (s)': deceleration_time,
        'Time to Mid Velocity (s)': time_to_mid_velocity,
        'Time to Max Deceleration (s)': time_to_max_deceleration,
        'Deceleration Distance (m)': deceleration_distance,
        'Average Horizontal Braking Force (N)': avg_horizontal_braking_force,
        'Early Average Horizontal Braking Force (N)': early_avg_horizontal_braking_force,
        'Late Average Horizontal Braking Force (N)': late_avg_horizontal_braking_force,
        'Maximum Horizontal Braking Force (N)': max_horizontal_braking_force,
        'Average Horizontal Braking Power (W)': avg_horizontal_braking_power,
        'Early Average Horizontal Braking Power (W)': early_avg_horizontal_braking_power,
        'Late Average Horizontal Braking Power (W)': late_avg_horizontal_braking_power,
        'Maximum Horizontal Braking Power (W)': max_horizontal_braking_power,
        'Average Horizontal Braking Impulse (N·s)': avg_horizontal_braking_impulse,
        'Early Average Horizontal Braking Impulse (N·s)': early_avg_horizontal_braking_impulse,
        'Late Average Horizontal Braking Impulse (N·s)': late_avg_horizontal_braking_impulse,
        'Maximum Horizontal Braking Impulse (N·s)': max_horizontal_braking_impulse,
        'Half Max Velocity Mean Acceleration (m/s^2)': half_max_velocity_mean_acceleration,
        'Half Max Velocity Peak Acceleration (m/s^2)': half_max_velocity_peak_acceleration
    })

    # ... (previous code for creating graphs remains the same)

# Export the data for all players to a single CSV file
csv_filename = "all_players_deceleration_results.csv"
fieldnames = list(all_players_data[0].keys())

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_players_data)

print(f"Results exported to {csv_filename}")

