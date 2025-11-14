import * as d3 from "d3";
import { frequencyToNoteName, generateNoteTicks } from "../utils";

export const createAxes = (
  parentGroup,
  xScale,
  yScale,
  yDomain,
  feature,
  innerHeight,
  yLabel
) => {
  // Setup yAxis
  let yAxis;
  if (feature === "pitch") {
    const { noteTicks } = generateNoteTicks(yDomain);

    yAxis = d3
      .axisLeft(yScale)
      .tickValues(noteTicks)
      .tickFormat(frequencyToNoteName);
  } else {
    yAxis = d3.axisLeft(yScale).ticks(5);
  }

  const xAxisGroup = parentGroup
    .append("g")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(d3.axisBottom(xScale).ticks(0));

  const yAxisGroup = parentGroup.append("g").call(yAxis);

  // Style axes
  [xAxisGroup, yAxisGroup].forEach((axis) => {
    axis
      .selectAll("path, line")
      .attr("stroke", "#E0E0E0")
      .attr("stroke-opacity", 0.25);
  });

  yAxisGroup
    .selectAll("text")
    .attr("fill", "#E0E0E0")
    .attr("fill-opacity", 0.7);

  // Y-axis label
  parentGroup
    .append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", -35)
    .attr("x", -innerHeight / 2)
    .attr("text-anchor", "middle")
    .attr("fill", "#E0E0E0")
    .attr("opacity", 0.7)
    .text(yLabel);

  return { xAxisGroup, yAxisGroup };
};
