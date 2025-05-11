import { useRef, useEffect } from "react";
import * as d3 from "d3";

const LineGraph = ({
  data,
  width,
  height,
  xLabel,
  yLabel,
  highlightedSections = [],
  yMin,
  yMax,
}) => {
  const ref = useRef();

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const xScale = d3
      .scaleLinear()
      .domain([0, data.length - 1])
      .range([0, innerWidth]);

    const yExtent = d3.extent(data);
    const yDomain = [
      yMin !== undefined ? yMin : yExtent[0],
      yMax !== undefined ? yMax : yExtent[1],
    ];
    const yScale = d3.scaleLinear().domain(yDomain).range([innerHeight, 0]);

    const line = d3
      .line()
      .x((d, i) => xScale(i))
      .y((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    const area = d3
      .area()
      .x((d, i) => xScale(i))
      .y0(yScale(yDomain[0]))
      .y1((d) => yScale(d))
      .curve(d3.curveMonotoneX);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Gradient definition
    const defs = svg.append("defs");
    const gradient = defs
      .append("linearGradient")
      .attr("id", "line-gradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "0%")
      .attr("y2", "100%");

    gradient
      .append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#FF89BB")
      .attr("stop-opacity", 0.4);

    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#90F1EF")
      .attr("stop-opacity", 0);

    // Highlighted sections
    highlightedSections.forEach(({ start, end }) => {
      g.append("rect")
        .attr("x", xScale(start))
        .attr("y", 0)
        .attr("width", xScale(end) - xScale(start))
        .attr("height", innerHeight)
        .attr("fill", "#FFCB6B")
        .attr("opacity", 0.25);
    });

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(10);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    const xAxisGroup = g
      .append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);

    const yAxisGroup = g.append("g").call(yAxis);

    // Axis styling
    [xAxisGroup, yAxisGroup].forEach((axis) => {
      axis
        .selectAll("path, line")
        .attr("stroke", "#E0E0E0")
        .attr("stroke-opacity", 0.25); // Adjust the opacity for axis lines and paths

      axis.selectAll("text").attr("fill", "#E0E0E0").attr("fill-opacity", 0.7); // Adjust the opacity for axis labels
    });

    // Axis labels
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + 30)
      .attr("text-anchor", "middle")
      .attr("fill", "#E0E0E0")
      .attr("opacity", 0.7)
      .text(xLabel);

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -35)
      .attr("x", -innerHeight / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#E0E0E0")
      .attr("opacity", 0.7)
      .text(yLabel);

    // Area under line
    g.append("path")
      .datum(data)
      .attr("fill", "url(#line-gradient)")
      .attr("d", area);

    // Line
    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#FF89BB")
      .attr("stroke-width", 2)
      .attr("d", line);
  }, [data, width, height, highlightedSections, xLabel, yLabel, yMin, yMax]);

  return <svg ref={ref} width={width} height={height} />;
};

export default LineGraph;
