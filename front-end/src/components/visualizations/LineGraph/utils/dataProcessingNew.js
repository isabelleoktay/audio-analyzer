import * as d3 from "d3";

/**
 * Filters out values below a threshold, replacing them with null (for cleaner line gaps)
 */
export const filterDataBelowThresholdNew = (data, threshold) => {
  if (!Array.isArray(data)) return [];
  return data.map((d) => (d < threshold ? null : d));
};

/**
 * Validates data arrays before processing
 */
export const validateChartDataNew = (data) => {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return { isValid: false, error: "No valid data provided" };
  }
  return { isValid: true };
};

/**
 * Processes and normalizes feature data for rendering in OverlayLineGraph
 * Handles missing or empty reference datasets gracefully.
 */
export const processChartDataNew = (
  primaryData = [],
  secondaryData = [],
  yMin,
  yMax,
  innerHeight
) => {
  if (primaryData.length === 0) {
    console.warn("⚠️ processChartDataNew: No primary data provided.");
  }

  // --- Compute Y-domain from whichever arrays have data ---
  const combinedData =
    primaryData.length > 0 || secondaryData.length > 0
      ? [...primaryData, ...secondaryData]
      : [0, 1]; // prevent crash if both empty

  const yExtent = d3.extent(combinedData);
  const yDomain = [
    yMin !== undefined ? yMin : yExtent[0],
    yMax !== undefined ? yMax : yExtent[1],
  ];

  // --- Filter out invalid / below-threshold points ---
  const filteredPrimaryData = filterDataBelowThresholdNew(
    primaryData,
    yDomain[0]
  );
  const filteredSecondaryData = filterDataBelowThresholdNew(
    secondaryData,
    yDomain[0]
  );

  // --- Always return a valid scale ---
  const yScale = d3.scaleLinear().domain(yDomain).range([innerHeight, 0]);

  return {
    isValid: true,
    filteredPrimaryData,
    filteredSecondaryData,
    yDomain,
    yScale,
  };
};
