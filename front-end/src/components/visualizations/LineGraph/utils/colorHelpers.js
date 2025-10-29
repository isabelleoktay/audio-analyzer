export const getFeatureColorStops = (feature) => {
  const colorMaps = {
    extents: [
      { value: Infinity, color: "#FFCB6B" },
      { value: 7, color: "#7aff6b" },
      { value: 2, color: "#7aff6b" },
      { value: -Infinity, color: "#FFCB6B" },
    ],
    rates: [
      { value: Infinity, color: "#ff6b6b" },
      { value: 10, color: "#ff6b6b" },
      { value: 9, color: "#ffb36b" },
      { value: 7, color: "#FFCB6B" },
      { value: 5, color: "#7aff6b" },
      { value: 3, color: "#FFCB6B" },
      { value: 2, color: "#ffb36b" },
      { value: 0, color: "#ff6b6b" },
      { value: -Infinity, color: "#ff6b6b" },
    ],
  };

  return colorMaps[feature] || [];
};

export function getDefaultLineGradientStops(lineColor = "#FF89BB") {
  return [
    { offset: "0%", color: lineColor, opacity: 0.8 },
    { offset: "40%", color: lineColor, opacity: 0.4 },
    { offset: "100%", color: lineColor, opacity: 0 },
  ];
}

export const getTooltipColors = (isSilence) => ({
  bgColor: isSilence ? "#E0E0E0" : "#FF89BB",
  textColor: isSilence ? "#1E1E2F" : "#E0E0E0",
  opacity: 0.5,
});
