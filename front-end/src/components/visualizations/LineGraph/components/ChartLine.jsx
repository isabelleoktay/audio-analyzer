import { createLineGenerator, createAreaGenerator } from "../utils";

/**
 * Draws the main and optional reference datasets on the chart.
 * Handles missing or empty secondary data gracefully.
 */
export const createMainChart = (
  chartGroup,
  filteredPrimaryData = [],
  primaryLineColor = "#FF89BB",
  filteredSecondaryData = [],
  secondaryLineColor = "#CCCCCC",
  xScale,
  yScale,
  yDomain
) => {
  const line = createLineGenerator(xScale, yScale);
  const area = createAreaGenerator(xScale, yScale, yDomain?.[0] ?? 0);

  // --- Draw reference dataset first (behind main) ---
  let secondaryDataAreaPath = null;
  let secondaryDataLinePath = null;

  if (filteredSecondaryData.length > 0) {
    try {
      secondaryDataAreaPath = chartGroup
        .append("path")
        .datum(filteredSecondaryData)
        .attr("class", "ref-area")
        .attr("fill", "url(#secondary-line-gradient)")
        .attr("opacity", 0.9)
        .attr("d", area);

      secondaryDataLinePath = chartGroup
        .append("path")
        .datum(filteredSecondaryData)
        .attr("class", "ref-line")
        .attr("fill", "none")
        .attr("stroke", secondaryLineColor)
        .attr("stroke-width", 2)
        .attr("d", line);
    } catch (err) {
      console.warn("Failed to draw secondary line:", err);
    }
  }

  // --- Draw main dataset (in front) ---;
  const areaPath = chartGroup
    .append("path")
    .datum(filteredPrimaryData)
    .attr("class", "main-area")
    .attr("fill", "url(#line-gradient)")
    .attr("opacity", 0.9)
    .attr("d", area);

  const linePath = chartGroup
    .append("path")
    .datum(filteredPrimaryData)
    .attr("class", "main-line")
    .attr("fill", "none")
    .attr("stroke", primaryLineColor)
    .attr("stroke-width", 2)
    .attr("d", line);

  return { areaPath, linePath, secondaryDataAreaPath, secondaryDataLinePath };
};

/**
 * Updates the chart when zooming, resizing, or re-rendering.
 * Handles absent reference datasets safely.
 */
export const updateMainChart = (
  chartGroup,
  filteredPrimaryData,
  filteredSecondaryData = null,
  newXScale,
  currentYScale,
  yDomain
) => {
  const line = createLineGenerator(newXScale, currentYScale);
  const area = createAreaGenerator(newXScale, currentYScale, yDomain?.[0]);

  // --- Update main line and area ---
  chartGroup.select(".main-area").datum(filteredPrimaryData).attr("d", area);
  chartGroup.select(".main-line").datum(filteredPrimaryData).attr("d", line);

  // --- Update secondary/reference paths if they exist ---
  if (filteredSecondaryData && filteredSecondaryData.length > 0) {
    chartGroup.select(".ref-area").datum(filteredSecondaryData).attr("d", area);
    chartGroup.select(".ref-line").datum(filteredSecondaryData).attr("d", line);
  }
};
