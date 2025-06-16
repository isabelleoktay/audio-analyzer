const frequencyToNote = (frequency) => {
  const A4 = 440; // Frequency of A4
  const noteNames = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
  ];

  if (frequency <= 0) return "";

  const semitonesFromA4 = Math.round(12 * Math.log2(frequency / A4));
  const octave = Math.floor((semitonesFromA4 + 9) / 12) + 4; // Octave calculation
  const noteIndex = (semitonesFromA4 + 9) % 12; // Note index in the `noteNames` array

  return `${noteNames[(noteIndex + 12) % 12]}${octave}`;
};

export default frequencyToNote;
