import { createLineGenerator, createAreaGenerator } from "../utils";

export const createMainChart = (
  chartGroup,
  filteredData,
  xScale,
  yScale,
  yDomain,
  lineColor
) => {
  const line = createLineGenerator(xScale, yScale);
  const area = createAreaGenerator(xScale, yScale, yDomain[0]);

  // Draw main area
  const areaPath = chartGroup
    .append("path")
    .datum(filteredData)
    .attr("class", "main-area")
    .attr("fill", "url(#line-gradient)")
    .attr("d", area);

  // Draw main line
  const linePath = chartGroup
    .append("path")
    .datum(filteredData)
    .attr("class", "main-line")
    .attr("fill", "none")
    .attr("stroke", lineColor)
    .attr("stroke-width", 2)
    .attr("d", line);

  return { areaPath, linePath };
};

export const updateMainChart = (
  chartGroup,
  filteredData,
  newXScale,
  currentYScale,
  yDomain
) => {
  const line = createLineGenerator(newXScale, currentYScale);
  const area = createAreaGenerator(newXScale, currentYScale, yDomain[0]);

  // Update main line and area
  chartGroup.select(".main-area").datum(filteredData).attr("d", area);
  chartGroup.select(".main-line").datum(filteredData).attr("d", line);
};
