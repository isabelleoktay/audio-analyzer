export function createVerticalBackgroundGradient({
  svgDefs,
  id,
  yDomain,
  colorStops,
}) {
  const [yMin, yMax] = yDomain;

  // Validate inputs
  if (!svgDefs || !id || !yDomain || !colorStops) {
    console.warn(
      "Missing required parameters for createVerticalBackgroundGradient"
    );
    return null;
  }

  if (yMin === undefined || yMax === undefined || isNaN(yMin) || isNaN(yMax)) {
    console.warn("Invalid yDomain values:", { yMin, yMax });
    return null;
  }

  // Converts a data y-value to a percent offset in gradient space
  const getOffset = (value) => {
    // Handle invalid values
    if (value === undefined || value === null || isNaN(value)) {
      console.warn("Invalid value for gradient offset:", value);
      return 0;
    }

    // Handle case where yMin equals yMax (no range)
    if (yMax === yMin) {
      return 50; // Put all stops in the middle
    }

    const offset = ((yMax - value) / (yMax - yMin)) * 100;
    const clampedOffset = Math.max(0, Math.min(100, offset));

    // Final check for NaN
    if (isNaN(clampedOffset)) {
      console.warn("Calculated offset is NaN:", { value, yMin, yMax, offset });
      return 0;
    }

    return clampedOffset;
  };

  const gradient = svgDefs
    .append("linearGradient")
    .attr("id", id)
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  colorStops.forEach(({ value, color }) => {
    const offset = getOffset(value);
    gradient
      .append("stop")
      .attr("offset", `${offset}%`)
      .attr("stop-color", color);
  });

  return `url(#${id})`;
}
