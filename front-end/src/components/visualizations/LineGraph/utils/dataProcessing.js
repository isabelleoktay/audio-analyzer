import * as d3 from "d3";

export const filterDataBelowThreshold = (data, threshold) => {
  return data.map((d) => (d < threshold ? null : d));
};

export const validateChartData = (data) => {
  if (!data || data.length === 0) {
    return { isValid: false, error: "No data provided" };
  }

  if (!Array.isArray(data)) {
    return { isValid: false, error: "Data must be an array" };
  }

  return { isValid: true };
};

export const processChartData = (data, yMin, yMax, innerHeight) => {
  const validation = validateChartData(data);
  if (!validation.isValid) {
    return { isValid: false, error: validation.error };
  }

  const yExtent = d3.extent(data);
  const yDomain = [
    yMin !== undefined ? yMin : yExtent[0],
    yMax !== undefined ? yMax : yExtent[1],
  ];

  const filteredData = filterDataBelowThreshold(data, yDomain[0]);

  // Create the y scale here too
  const yScale = d3.scaleLinear().domain(yDomain).range([innerHeight, 0]);

  return {
    isValid: true,
    originalData: data,
    filteredData,
    yExtent,
    yDomain,
    yScale, // Include the scale
  };
};
