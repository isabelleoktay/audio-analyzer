// src/mock/mockAudioFeatures.js

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

const mockInputFeatures = {
  pitch: {
    audioUrl: inputAudioURL,
    duration: 12.3, // seconds
    data: [{ label: "pitch", data: generateRandomData(200, 70, 140) }],
  },
  dynamics: {
    audioUrl: inputAudioURL,
    duration: 12.3,
    data: [{ label: "dynamics", data: generateRandomData(200, -15, 5) }],
  },
  tempo: {
    audioUrl: inputAudioURL,
    duration: 12.3,
    data: [{ label: "tempo", data: generateRandomData(200, 110, 130) }],
  },
  phonation: {
    audioUrl: inputAudioURL,
    duration: 12.3,
    data: [
      { label: "breathy", data: generateRandomData(200, 0, 1) },
      { label: "neutral", data: generateRandomData(200, 0.3, 0.9) },
      { label: "flow", data: generateRandomData(200, 0.1, 0.5) },
      { label: "pressed", data: generateRandomData(200, 0.4, 0.8) },
    ],
  },

  "pitch mod.": {
    CLAP: {
      audioUrl: inputAudioURL,
      duration: 12.3,
      data: [
        { label: "vibrato", data: generateRandomData(200, 0.6, 0.9) },
        { label: "trill", data: generateRandomData(200, 0.1, 0.7) },
        { label: "trillo", data: generateRandomData(200, 0.3, 0.4) },
        { label: "straight", data: generateRandomData(200, 0.3, 1.0) },
      ],
    },
    Whisper: {
      audioUrl: inputAudioURL,
      duration: 12.3,
      data: [
        { label: "vibrato", data: generateRandomData(200, 0.65, 0.95) },
        { label: "trill", data: generateRandomData(200, 0.15, 0.6) },
        { label: "trillo", data: generateRandomData(200, 0.35, 0.5) },
        { label: "straight", data: generateRandomData(200, 0.25, 0.9) },
      ],
    },
  },

  "vocal tone": {
    CLAP: {
      audioUrl: inputAudioURL,
      duration: 12.3,
      data: [
        { label: "spoken", data: generateRandomData(200, 0.3, 1.0) },
        { label: "inhaled", data: generateRandomData(200, 0.2, 1.0) },
        { label: "belt", data: generateRandomData(200, 0.3, 0.9) },
        { label: "breathy", data: generateRandomData(200, 0.4, 1.0) },
        { label: "vocal fry", data: generateRandomData(200, 0.1, 1.0) },
      ],
    },
    Whisper: {
      audioUrl: inputAudioURL,
      duration: 12.3,
      data: [
        { label: "spoken", data: generateRandomData(200, 0.25, 0.95) },
        { label: "inhaled", data: generateRandomData(200, 0.15, 0.9) },
        { label: "belt", data: generateRandomData(200, 0.3, 1.0) },
        { label: "breathy", data: generateRandomData(200, 0.35, 95) },
        { label: "vocal fry", data: generateRandomData(200, 0.7, 0.9) },
      ],
    },
  },
};

const mockReferenceFeatures = {
  pitch: {
    audioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "pitch", data: generateRandomData(200, 69, 120) }],
  },
  dynamics: {
    audioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "dynamics", data: generateRandomData(200, -15, 5) }],
  },
  tempo: {
    audioUrl: referenceAudioURL,
    duration: 12.3,
    data: [{ label: "tempo", data: generateRandomData(200, 120, 140) }],
  },
  phonation: {
    audioUrl: referenceAudioURL,
    duration: 12.3,
    data: [
      { label: "breathy", data: generateRandomData(200, 0, 0.9) },
      { label: "neutral", data: generateRandomData(200, 0.2, 0.8) },
      { label: "flow", data: generateRandomData(200, 0.2, 0.6) },
      { label: "pressed", data: generateRandomData(200, 0.5, 0.9) },
    ],
  },

  "pitch mod.": {
    CLAP: {
      audioUrl: referenceAudioURL,
      duration: 12.3,
      data: [
        { label: "vibrato", data: generateRandomData(200, 0.6, 0.9) },
        { label: "trill", data: generateRandomData(200, 0.1, 0.7) },
        { label: "trillo", data: generateRandomData(200, 0.3, 0.4) },
        { label: "straight", data: generateRandomData(200, 0.3, 0.5) },
      ],
    },
    Whisper: {
      audioUrl: referenceAudioURL,
      duration: 12.3,
      data: [
        { label: "vibrato", data: generateRandomData(200, 0.65, 0.95) },
        { label: "trill", data: generateRandomData(200, 0.15, 0.6) },
        { label: "trillo", data: generateRandomData(200, 0.35, 0.5) },
        { label: "straight", data: generateRandomData(200, 0.25, 0.9) },
      ],
    },
  },

  "vocal tone": {
    CLAP: {
      audioUrl: referenceAudioURL,
      duration: 12.3,
      data: [
        { label: "spoken", data: generateRandomData(200, 0.3, 1.0) },
        { label: "inhaled", data: generateRandomData(200, 0.2, 1.0) },
        { label: "belt", data: generateRandomData(200, 0.1, 1.0) },
        { label: "breathy", data: generateRandomData(200, 0.4, 1.0) },
        { label: "vocal fry", data: generateRandomData(200, 0.1, 1.0) },
      ],
    },
    Whisper: {
      audioUrl: referenceAudioURL,
      duration: 12.3,
      data: [
        { label: "spoken", data: generateRandomData(200, 0.25, 0.95) },
        { label: "inhaled", data: generateRandomData(200, 0.15, 0.9) },
        { label: "belt", data: generateRandomData(200, 0.12, 1.0) },
        { label: "breathy", data: generateRandomData(200, 0.35, 0.95) },
        { label: "vocal fry", data: generateRandomData(200, 0.7, 0.9) },
      ],
    },
  },
};

export { mockInputFeatures, mockReferenceFeatures };
