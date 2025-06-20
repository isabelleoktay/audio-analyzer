export const frequencyToNoteName = (frequency) => {
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
  const noteNum = Math.round(12 * Math.log2(frequency / 440) + 69);
  const octave = Math.floor(noteNum / 12) - 1;
  return noteNames[noteNum % 12] + octave;
};

export const generateNoteTicks = (yDomain) => {
  const effectiveYMin = yDomain[0] > 0 ? yDomain[0] : 20;
  const lowerNote = Math.ceil(12 * Math.log2(effectiveYMin / 440) + 69);
  const upperNote = Math.floor(12 * Math.log2(yDomain[1] / 440) + 69);

  const noteTicks = [];
  for (let n = lowerNote; n <= upperNote; n++) {
    const tickFreq = 440 * Math.pow(2, (n - 69) / 12);
    if (tickFreq >= yDomain[0] && tickFreq <= yDomain[1]) {
      noteTicks.push(tickFreq);
    }
  }

  return { noteTicks, lowerNote, upperNote };
};

export const generateNoteStripeData = (yDomain) => {
  const effectiveYMin = yDomain[0] > 0 ? yDomain[0] : 20;
  const lowerNote = Math.ceil(12 * Math.log2(effectiveYMin / 440) + 69);
  const upperNote = Math.floor(12 * Math.log2(yDomain[1] / 440) + 69);

  const stripes = [];
  for (let n = lowerNote; n <= upperNote; n++) {
    const f = 440 * Math.pow(2, (n - 69) / 12);
    const nextF = 440 * Math.pow(2, (n - 68) / 12);
    const regionStart = Math.max(f, yDomain[0]);
    const regionEnd = Math.min(nextF, yDomain[1]);
    const color = n % 2 === 0 ? "#1E1E2F" : "#5F5F95";

    stripes.push({
      start: regionStart,
      end: regionEnd,
      color,
      opacity: 0.25,
    });
  }

  return stripes;
};
