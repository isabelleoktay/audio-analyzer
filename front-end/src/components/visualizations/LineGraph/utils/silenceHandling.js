export const detectSilenceRanges = (filteredData) => {
  const silenceRanges = [];
  let silenceStart = null;

  filteredData.forEach((d, i) => {
    if (d === null && silenceStart === null) {
      silenceStart = i;
    } else if (d !== null && silenceStart !== null) {
      silenceRanges.push({ start: silenceStart, end: i - 1 });
      silenceStart = null;
    }
  });

  if (silenceStart !== null) {
    silenceRanges.push({
      start: silenceStart,
      end: filteredData.length - 1,
    });
  }

  return silenceRanges;
};

export const createSilenceLineData = (
  silenceRanges,
  xScale,
  innerHeight,
  minWidth = 5
) => {
  return silenceRanges
    .map(({ start, end }) => {
      const width = xScale(end) - xScale(start);
      if (width > minWidth) {
        return {
          x1: xScale(start),
          x2: xScale(end),
          y1: innerHeight - 5,
          y2: innerHeight - 5,
          stroke: "#E0E0E0",
          strokeWidth: 2,
          strokeDasharray: "3,3",
          opacity: 0.7,
        };
      }
      return null;
    })
    .filter(Boolean);
};
