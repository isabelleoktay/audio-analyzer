import * as d3 from "d3";

export const createXScale = (data, xLabels, innerWidth) => {
  const hasXLabels = Array.isArray(xLabels) && xLabels.length > 0;
  const xDomainMax = hasXLabels ? xLabels.length - 1 : data.length - 1;

  return d3.scaleLinear().domain([0, xDomainMax]).range([0, innerWidth]);
};

export const createYScale = (data, yMin, yMax, innerHeight) => {
  const yExtent = d3.extent(data);
  const yDomain = [
    yMin !== undefined ? yMin : yExtent[0],
    yMax !== undefined ? yMax : yExtent[1],
  ];

  return {
    scale: d3.scaleLinear().domain(yDomain).range([innerHeight, 0]),
    domain: yDomain,
  };
};

export const createZoomedXScale = (originalScale, selection) => {
  const [x0, x1] = selection;
  const newDomain = [originalScale.invert(x0), originalScale.invert(x1)];

  return d3.scaleLinear().domain(newDomain).range(originalScale.range());
};
