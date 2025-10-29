import { createLineGenerator, createAreaGenerator } from "../utils";

/**
 * Draws the main and optional reference datasets on the chart.
 * Handles missing or empty secondary data gracefully.
 */
export const createMainChartNew = (
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
        .attr("fill", "none")
        .attr("opacity", 0.2)
        .attr("d", area);

      secondaryDataLinePath = chartGroup
        .append("path")
        .datum(filteredSecondaryData)
        .attr("class", "ref-line")
        .attr("fill", "none")
        .attr("stroke", secondaryLineColor)
        .attr("stroke-width", 2)
        .attr("opacity", 0.7)
        .attr("d", line);
    } catch (err) {
      console.warn("⚠️ Failed to draw secondary line:", err);
    }
  }

  // --- Draw main dataset (in front) ---
  let areaPath = null;
  let linePath = null;

  if (filteredPrimaryData.length > 0) {
    areaPath = chartGroup
      .append("path")
      .datum(filteredPrimaryData)
      .attr("class", "main-area")
      .attr("fill", "url(#line-gradient)")
      .attr("opacity", 0.3)
      .attr("d", area);

    linePath = chartGroup
      .append("path")
      .datum(filteredPrimaryData)
      .attr("class", "main-line")
      .attr("fill", "none")
      .attr("stroke", primaryLineColor)
      .attr("stroke-width", 2)
      .attr("d", line);
  } else {
    console.warn("⚠️ createMainChartNew: No primary data to render.");
  }

  return { areaPath, linePath, secondaryDataAreaPath, secondaryDataLinePath };
};

/**
 * Updates the chart when zooming, resizing, or re-rendering.
 * Handles absent reference datasets safely.
 */
export const updateMainChartNew = (
  chartGroup,
  filteredPrimaryData = [],
  filteredSecondaryData = [],
  newXScale,
  currentYScale,
  yDomain
) => {
  const line = createLineGenerator(newXScale, currentYScale);
  const area = createAreaGenerator(newXScale, currentYScale, yDomain?.[0] ?? 0);

  // --- Update main data paths (only if they exist) ---
  const mainArea = chartGroup.select(".main-area");
  const mainLine = chartGroup.select(".main-line");

  if (!mainArea.empty() && filteredPrimaryData.length > 0) {
    mainArea.datum(filteredPrimaryData).attr("d", area);
  }

  if (!mainLine.empty() && filteredPrimaryData.length > 0) {
    mainLine.datum(filteredPrimaryData).attr("d", line);
  }

  // --- Update secondary/reference paths if they exist ---
  const secondaryArea = chartGroup.select(".ref-area");
  const secondaryLine = chartGroup.select(".ref-line");

  if (!secondaryArea.empty() && filteredSecondaryData.length > 0) {
    secondaryArea.datum(filteredSecondaryData).attr("d", area);
  }

  if (!secondaryLine.empty() && filteredSecondaryData.length > 0) {
    secondaryLine.datum(filteredSecondaryData).attr("d", line);
  }
};
