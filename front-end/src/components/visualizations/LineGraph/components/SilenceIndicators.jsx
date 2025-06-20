import { detectSilenceRanges } from "../utils";

export const createSilenceIndicators = (
  chartGroup,
  filteredData,
  xScale,
  innerHeight
) => {
  const silenceRanges = detectSilenceRanges(filteredData);

  chartGroup.selectAll(".silence-line").remove();
  silenceRanges.forEach(({ start, end }) => {
    const width = xScale(end) - xScale(start);
    if (width > 5) {
      chartGroup
        .append("line")
        .attr("class", "silence-line")
        .attr("x1", xScale(start))
        .attr("x2", xScale(end))
        .attr("y1", innerHeight - 5)
        .attr("y2", innerHeight - 5)
        .attr("stroke", "#E0E0E0")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "3,3")
        .attr("opacity", 0.7);
    }
  });
};

export const updateSilenceIndicators = (
  chartGroup,
  filteredData,
  newXScale,
  innerHeight
) => {
  const silenceRanges = detectSilenceRanges(filteredData);

  chartGroup.selectAll(".silence-line").remove();
  silenceRanges.forEach(({ start, end }) => {
    const width = newXScale(end) - newXScale(start);
    if (width > 5) {
      chartGroup
        .append("line")
        .attr("class", "silence-line")
        .attr("x1", newXScale(start))
        .attr("x2", newXScale(end))
        .attr("y1", innerHeight - 5)
        .attr("y2", innerHeight - 5)
        .attr("stroke", "#E0E0E0")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "3,3")
        .attr("opacity", 0.7);
    }
  });
};
