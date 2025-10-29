// src/mock/mockAudioFeatures.js

// Utility functions for generating pseudo-random data
const generateWaveData = (
  numPoints,
  base = 1,
  amplitude = 1,
  frequency = 0.05
) =>
  Array.from(
    { length: numPoints },
    (_, i) =>
      base +
      amplitude * Math.sin(i * frequency) +
      Math.random() * amplitude * 0.2
  );

const generateRandomData = (numPoints, min = 0, max = 1, smoothness = 0.08) => {
  const data = [];
  let value = min + Math.random() * (max - min); // start point

  for (let i = 0; i < numPoints; i++) {
    // Small random step for smooth transitions
    const step = (Math.random() - 0.5) * (max - min) * smoothness;
    value += step;

    // Clamp to range
    if (value < min) value = min;
    if (value > max) value = max;

    data.push(value);
  }
  return data;
};

// Mock audio file URLs (you can replace with actual local audio paths)
const inputAudioURL = "front-end/public/audio/development/input.wav";
const referenceAudioURL = "front-end/public/audio/development/reference.wav";

const mockInputAudioFeatures = {
  pitch: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3, // seconds
    data: [{ label: "pitch", data: generateWaveData(200, 220, 40, 0.07) }],
  },
  dynamics: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "dynamics", data: generateWaveData(200, -15, 5, 0.04) }],
  },
  tempo: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "tempo", data: generateWaveData(200, 110, 10, 0.02) }],
  },
  phonation: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "breathy", data: generateRandomData(200, 0, 100) },
      { label: "neutral", data: generateRandomData(200, 30, 90) },
      { label: "flow", data: generateRandomData(200, 10, 50) },
      { label: "pressed", data: generateRandomData(200, 40, 80) },
    ],
  },

  // === Pitch modulation ===
  "pitch mod.": {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "vibrato", data: generateRandomData(200, 50, 100) },
      { label: "trill", data: generateRandomData(200, 20, 80) },
      { label: "trillo", data: generateRandomData(200, 40, 50) },
      { label: "straight", data: generateRandomData(200, 20, 90) },
    ],
  },

  // === Vocal tone ===
  "vocal tone": {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "spoken", data: generateRandomData(200, 30, 100) },
      { label: "inhaled", data: generateRandomData(200, 20, 100) },
      { label: "belt", data: generateRandomData(200, 10, 100) },
      { label: "breathy", data: generateRandomData(200, 40, 100) },
      { label: "vocal fry", data: generateRandomData(200, 10, 100) },
    ],
  },
};

const mockReferenceAudioFeatures = {
  pitch: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3, // seconds
    data: [{ label: "pitch", data: generateWaveData(200, 220, 40, 0.07) }],
  },
  dynamics: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "dynamics", data: generateWaveData(200, -15, 5, 0.04) }],
  },
  tempo: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "tempo", data: generateWaveData(200, 110, 10, 0.02) }],
  },
  phonation: {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "breathy", data: generateRandomData(200, 0, 90) },
      { label: "neutral", data: generateRandomData(200, 20, 80) },
      { label: "flow", data: generateRandomData(200, 20, 60) },
      { label: "pressed", data: generateRandomData(200, 50, 90) },
    ],
  },

  // === Pitch modulation ===
  "pitch mod.": {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "vibrato", data: generateRandomData(200, 60, 90) },
      { label: "trill", data: generateRandomData(200, 10, 70) },
      { label: "trillo", data: generateRandomData(200, 30, 40) },
      { label: "straight", data: generateRandomData(200, 30, 100) },
    ],
  },

  // === Vocal tone ===
  "vocal tone": {
    audioUrl: inputAudioURL,
    referenceAudioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "spoken", data: generateRandomData(200, 30, 100) },
      { label: "inhaled", data: generateRandomData(200, 20, 100) },
      { label: "belt", data: generateRandomData(200, 10, 100) },
      { label: "breathy", data: generateRandomData(200, 40, 100) },
      { label: "vocal fry", data: generateRandomData(200, 10, 100) },
    ],
  },
};

export {
  mockInputAudioFeatures as mockInputFeatures,
  mockReferenceAudioFeatures as mockReferenceFeatures,
};
