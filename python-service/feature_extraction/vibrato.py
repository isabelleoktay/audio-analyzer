import numpy as np
from feature_extraction.vibrato_utils import (
    find_stable_sections,
    is_oscillating,
    calculate_vibrato_extent,
    calculate_vibrato_rate,
)   

def extract_vibrato(pitches, smoothed_pitches, pitch_notes, min_sustained_length=5):
    # Identify where the pitch is stable -- pitch notes values are the same
    stable_sections = find_stable_sections(pitch_notes, min_sustained_length)
    vibrato_data = [0] * len(pitches)

    # Iterate through each stable section and check for oscillation
    for start, end in stable_sections:
        section_data = pitches[start:end + 1]
        smoothed_data = smoothed_pitches[start:end + 1]

        # Subtract the smoothed (DC) component to isolate the oscillations
        oscillation_data = section_data - smoothed_data

        # Only keep the section if it's oscillating
        if is_oscillating(oscillation_data, smoothed_pitches):
            for i in range(start, end + 1):
                if pitches[i] == 0:
                    vibrato_data[i] = 0
                else:
                    vibrato_data[i] = oscillation_data[i - start]  # Store oscillations around zero

    return vibrato_data

def calculate_vibrato_parameters(vibrato_data, times, sr):
    vibrato_data = np.array(vibrato_data)
    times = np.array(times)

    vibrato_extents = np.zeros_like(vibrato_data, dtype=float)
    vibrato_rates = np.zeros_like(vibrato_data, dtype=float)

    # Identify sections with non-zero vibrato data
    non_zero_sections = np.where(vibrato_data != 0)[0]
    if len(non_zero_sections) == 0:
        return vibrato_extents, vibrato_rates  # No vibrato data present

    # Split into contiguous vibrato sections separated by zeros
    section_indices = np.split(non_zero_sections, np.where(np.diff(non_zero_sections) > 5)[0] + 1)

    # i=0
    for section in section_indices:
        section_data = vibrato_data[section]
        section_times = times[section]

        # Calculate zero crossings for frequency
        zero_crossings = np.where(np.diff(np.sign(section_data)))[0]
        if len(zero_crossings) < 2:
            continue  # Skip sections without full oscillations

        # Calculate vibrato extent
        calculate_vibrato_extent(vibrato_extents, zero_crossings, section_data, section)

        # Calculate vibrato rate
        calculate_vibrato_rate(vibrato_rates, sr, section_data, section_times, section)

    return vibrato_extents, vibrato_rates

def get_sections_with_vibrato(vibrato_extents, vibrato_rates, times, hop_duration_sec):
    indices = [i for i, extent in enumerate(vibrato_extents) if extent > 1.9]
    # Group consecutive indices into sections
    highlight_sections = []
    for idx in indices:
        if not highlight_sections or idx != highlight_sections[-1][-1] + 1:
            highlight_sections.append([idx])
        else:
            highlight_sections[-1].append(idx)
    
    # Calculate section metrics
    highlight_data = {
        'data': [],
        'audio': []
    }

    for section in highlight_sections:
        section_times = [times[i] for i in section]
        section_extents = [vibrato_extents[i] for i in section]
        section_rates = [vibrato_rates[i] for i in section]

        avg_extent = np.mean(section_extents)
        avg_rate = np.mean(section_rates)

        highlight_data["audio"].append({
            "start": section_times[0],
            "end": section_times[-1],
        })
        highlight_data["data"].append({
            "start": int(section_times[0] / hop_duration_sec),
            "end": int(section_times[-1] / hop_duration_sec),
        })

    return highlight_data