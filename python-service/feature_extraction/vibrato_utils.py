from config import LOOKUP_TABLE

import numpy as np
import pandas as pd

def get_note_for_frequency(frequency, file_path="lookup_table.csv"):
    # Load the lookup table
    lookup_table = pd.read_csv(file_path)

    if frequency == 0:
        return "Rest"

    # Ensure the Frequency column is numeric
    lookup_table['Frequency'] = pd.to_numeric(lookup_table['Frequency'], errors='coerce')

    # Ensure frequency falls within the lookup table range
    if frequency < lookup_table['Frequency'].min() or frequency > lookup_table['Frequency'].max():
        return 0

    # Find the closest frequency
    lookup_table['difference'] = (lookup_table['Frequency'] - frequency).abs()
    closest_row = lookup_table.loc[lookup_table['difference'].idxmin()]

    return closest_row['Note']

def frequencies_to_piano_notes(smoothed_pitches):
    """
    Convert an array of pitch frequencies into an array of corresponding piano notes.
    :param frequencies: List of pitch frequencies (in Hz).
    :return: List of corresponding piano notes (e.g., C1, C#3, B5, etc.).
    """

    # Define the reference frequency for A4 (440 Hz)
    pitch_notes = [
        get_note_for_frequency(pitch, file_path=LOOKUP_TABLE)
        if pitch > 0 else "Rest"
        for pitch in smoothed_pitches
    ]

    return pitch_notes


def is_oscillating(frequencies, smoothed_pitches, min_change_percentage=0.5):
    """
    Check if the section oscillates by comparing each frequency with the previous one.
    Oscillation is considered if there are more than two full oscillations (i.e., more than 2 up-down or down-up transitions),
    and each change in direction must exceed `min_change_percentage` of the DC value (smoothed average) of the section.
    """
    direction = None  # 'up' or 'down' or None for the first comparison
    oscillation_count = 0  # To count the number of full oscillations

    # Calculate the DC (mean) value of the smoothed pitches (the baseline)
    dc_value = sum(smoothed_pitches) / len(smoothed_pitches)

    # Calculate the minimum change required based on the DC value (e.g., 2% of the DC value)
    min_change = (min_change_percentage / 100) * dc_value

    for i in range(1, len(frequencies)):
        # Ensure the frequency change is greater than the calculated threshold (min_change)
        if abs(frequencies[i] - frequencies[i - 1]) > min_change:
            if frequencies[i] > frequencies[i - 1]:  # Going up
                if direction == 'down':  # If it was previously going down and now goes up
                    oscillation_count += 1  # Full oscillation detected
                    if oscillation_count > 2:  # More than 2 full oscillations
                        return True
                direction = 'up'
            elif frequencies[i] < frequencies[i - 1]:  # Going down
                if direction == 'up':  # If it was previously going up and now goes down
                    oscillation_count += 1  # Full oscillation detected
                    if oscillation_count > 2:  # More than 2 full oscillations
                        return True
                direction = 'down'

    return False  # If no full oscillation count greater than 2 with sufficient change was found

def find_stable_sections(pitch_notes, min_sustained_length_samples=10):
    stable_sections = []
    start = 0

    # Find stable sections based on pitch change
    for i in range(1, len(pitch_notes)):
        if pitch_notes[i] != pitch_notes[i - 1]:
            if i - start >= min_sustained_length_samples:
                # Only keep the section if it's non-zero
                if np.all(pitch_notes[start:i] != 'Rest'):
                    stable_sections.append((start, i - 1))
            start = i

    # Check the last section
    if len(pitch_notes) - start >= min_sustained_length_samples:
        if np.all(pitch_notes[start:] != 'Rest'):
            stable_sections.append((start, len(pitch_notes) - 1))

    return stable_sections

def calculate_vibrato_rate(vibrato_rates, sr, section_data, section_times, section):
    section_data = np.array(section_data)
    section_times = np.array(section_times)

    sign_changes = 0

    # Loop through the data and count the sign changes
    for i in range(1, len(section_data)):
        if (section_data[i-1] > 0 and section_data[i] < 0) or (section_data[i-1] < 0 and section_data[i] > 0):
            sign_changes += 1

    periods = sign_changes / 2

    total_time = section_times[-1] - section_times[0]

    if total_time > 0:
        frequency = periods / total_time
    else:
        frequency = 0

    vibrato_rates[section] = frequency

    return vibrato_rates

def calculate_vibrato_extent(vibrato_extents, zero_crossings, section_data, section):
    # Calculate vibrato extents (amplitude) for each cycle
    for i in range(len(zero_crossings) - 1):
        start_idx = zero_crossings[i]
        end_idx = zero_crossings[i + 1]
        cycle_data = section_data[start_idx:end_idx + 1]
        extent = (np.max(cycle_data) - np.min(cycle_data)) / 2
        vibrato_extents[section[start_idx:end_idx + 1]] = extent

    return vibrato_extents