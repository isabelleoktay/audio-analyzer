export const createHighlightedSections = (
  chartGroup,
  highlightedSections,
  xScale,
  innerHeight,
  feature
) => {
  highlightedSections.forEach((section) => {
    chartGroup
      .append("rect")
      .datum(section)
      .attr("class", "highlight-rect")
      .attr("x", xScale(section.start))
      .attr("y", 0)
      .attr("width", xScale(section.end) - xScale(section.start))
      .attr("height", innerHeight)
      .attr("fill", feature === "pitch" ? "#FF89BB" : "#FFCB6B") // Red for pitch, yellow otherwise
      .attr("opacity", 0.25);
  });
};

export const updateHighlightedSections = (chartGroup, newXScale) => {
  chartGroup
    .selectAll(".highlight-rect")
    .attr("x", (d) => newXScale(d.start))
    .attr("width", (d) => newXScale(d.end) - newXScale(d.start));
};
