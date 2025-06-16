export function createVerticalBackgroundGradient({
  svgDefs,
  id,
  yDomain,
  colorStops,
}) {
  const [yMin, yMax] = yDomain;

  // Converts a data y-value to a percent offset in gradient space
  const getOffset = (value) => {
    const offset = ((yMax - value) / (yMax - yMin)) * 100;
    return Math.max(0, Math.min(100, offset));
  };

  const gradient = svgDefs
    .append("linearGradient")
    .attr("id", id)
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  colorStops.forEach(({ value, color }) => {
    gradient
      .append("stop")
      .attr("offset", `${getOffset(value)}%`)
      .attr("stop-color", color);
  });

  return `url(#${id})`;
}
