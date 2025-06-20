import * as d3 from "d3";

export const createLineGenerator = (xScale, yScale) => {
  return d3
    .line()
    .x((d, i) => xScale(i))
    .y((d) => yScale(d))
    .curve(d3.curveMonotoneX)
    .defined((d) => d !== null);
};

export const createAreaGenerator = (xScale, yScale, yDomainMin) => {
  return d3
    .area()
    .x((d, i) => xScale(i))
    .y0(yScale(yDomainMin))
    .y1((d) => yScale(d))
    .curve(d3.curveMonotoneX)
    .defined((d) => d !== null);
};

export const createBrushWithHandler = (innerWidth, innerHeight, onBrushEnd) => {
  const brush = d3.brushX().extent([
    [0, 0],
    [innerWidth, innerHeight],
  ]);

  // Add the end handler that has access to the brush instance
  brush.on("end", function (event) {
    if (!event.selection) return;

    const [x0, x1] = event.selection;

    // Clear the brush selection first
    d3.select(this).call(brush.clear);

    // Then call the custom handler
    onBrushEnd(event, [x0, x1]);
  });

  return brush;
};

export const createClipPath = (defs, id, width, height) => {
  return defs
    .append("clipPath")
    .attr("id", id)
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", width)
    .attr("height", height);
};

export const createGradient = (defs, id, stops) => {
  const gradient = defs
    .append("linearGradient")
    .attr("id", id)
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "0%")
    .attr("y2", "100%");

  stops.forEach((stop) => {
    gradient
      .append("stop")
      .attr("offset", stop.offset)
      .attr("stop-color", stop.color)
      .attr("stop-opacity", stop.opacity);
  });

  return gradient;
};
