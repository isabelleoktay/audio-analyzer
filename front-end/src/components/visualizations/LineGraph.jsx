import { useRef, useEffect } from "react";
import * as d3 from "d3";

import { createVerticalBackgroundGradient } from "../../utils/createVerticalBackgroundGradient";

const LineGraph = ({
  feature = "pitch",
  data,
  width,
  height,
  xLabel,
  yLabel,
  highlightedSections = [],
  yMin,
  yMax,
  xLabels,
  lineColor = "#FF89BB",
}) => {
  const ref = useRef();

  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 20, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const hasXLabels = Array.isArray(xLabels) && xLabels.length > 0;
    const xDomainMax = hasXLabels ? xLabels.length - 1 : data.length - 1;
    const xScale = d3
      .scaleLinear()
      .domain([0, xDomainMax])
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

    const defs = svg.append("defs");
    // Gradient definition
    if (feature !== "rates" && feature !== "extents") {
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
    }

    // Highlighted sections (drawn below the graph)
    highlightedSections.forEach(({ start, end }) => {
      g.append("rect")
        .attr("x", xScale(start))
        .attr("y", 0)
        .attr("width", xScale(end) - xScale(start))
        .attr("height", innerHeight)
        .attr("fill", "#FFCB6B")
        .attr("opacity", 0.25);
    });

    // Setup yAxis
    let yAxis;
    if (feature === "pitch" || feature === "vibrato") {
      const frequencyToNoteName = (frequency) => {
        const noteNames = [
          "C",
          "C#",
          "D",
          "D#",
          "E",
          "F",
          "F#",
          "G",
          "G#",
          "A",
          "A#",
          "B",
        ];
        const noteNum = Math.round(12 * Math.log2(frequency / 440) + 69);
        const octave = Math.floor(noteNum / 12) - 1;
        return noteNames[noteNum % 12] + octave;
      };

      const effectiveYMin = yDomain[0] > 0 ? yDomain[0] : 20;
      const lowerNote = Math.ceil(12 * Math.log2(effectiveYMin / 440) + 69);
      const upperNote = Math.floor(12 * Math.log2(yDomain[1] / 440) + 69);

      const noteTicks = [];
      for (let n = lowerNote; n <= upperNote; n++) {
        const tickFreq = 440 * Math.pow(2, (n - 69) / 12);
        if (tickFreq >= yDomain[0] && tickFreq <= yDomain[1]) {
          noteTicks.push(tickFreq);
        }
      }

      yAxis = d3
        .axisLeft(yScale)
        .tickValues(noteTicks)
        .tickFormat(frequencyToNoteName);

      for (let n = lowerNote; n <= upperNote; n++) {
        const f = 440 * Math.pow(2, (n - 69) / 12);
        const nextF = 440 * Math.pow(2, (n - 68) / 12);
        const regionStart = Math.max(f, yDomain[0]);
        const regionEnd = Math.min(nextF, yDomain[1]);
        const yTop = yScale(regionEnd);
        const yBottom = yScale(regionStart);
        const stripeHeight = yBottom - yTop;
        const color = n % 2 === 0 ? "#1E1E2F" : "#5F5F95";
        g.append("rect")
          .attr("x", 0)
          .attr("y", yTop)
          .attr("width", innerWidth)
          .attr("height", stripeHeight)
          .attr("fill", color)
          .attr("opacity", 0.25)
          .lower();
      }
    } else {
      yAxis = d3.axisLeft(yScale).ticks(5);
    }

    if (feature === "extents" || feature === "rates") {
      const colorStops =
        feature === "extents"
          ? [
              { value: Infinity, color: "#FFCB6B" }, // Everything above 10
              { value: 7, color: "#7aff6b" },
              { value: 2, color: "#7aff6b" },
              { value: -Infinity, color: "#FFCB6B" },
            ]
          : [
              { value: Infinity, color: "#ff6b6b" }, // Everything above 10
              { value: 10, color: "#ff6b6b" },
              { value: 9, color: "#ffb36b" },
              { value: 7, color: "#FFCB6B" },
              { value: 5, color: "#7aff6b" },
              { value: 3, color: "#FFCB6B" },
              { value: 2, color: "#ffb36b" },
              { value: 0, color: "#ff6b6b" },
              { value: -Infinity, color: "#ff6b6b" }, // Everything below 0
            ];

      const gradientFill = createVerticalBackgroundGradient({
        svgDefs: defs,
        id: "extent-gradient",
        yDomain,
        colorStops,
        // colorStops: [
        //   { value: yMax, color: "yellow" },
        //   { value: 7, color: "yellow" },
        //   { value: 7, color: "green" },
        //   { value: 2, color: "green" },
        //   { value: 2, color: "yellow" },
        //   { value: yMin, color: "yellow" },
        // ],
      });

      g.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", innerWidth)
        .attr("height", innerHeight)
        .attr("fill", gradientFill)
        .attr("opacity", 0.25)
        .lower();
    }

    // if (feature === "rates" || feature === "extents") {
    //   const [yMin, yMax] = yDomain;

    //   // Clamp the band around y=6 only if it's within the visible domain
    //   const bandHalfSize = (yMax - yMin) / 6; // control how much color banding you want

    //   const rateGradient = defs
    //     .append("linearGradient")
    //     .attr("id", "rate-gradient")
    //     .attr("x1", "0%")
    //     .attr("y1", "0%")
    //     .attr("x2", "0%")
    //     .attr("y2", "100%");

    //   const getOffset = (value) => {
    //     // Clamp offset to [0, 100]
    //     return Math.max(
    //       0,
    //       Math.min(100, ((yMax - value) / (yMax - yMin)) * 100)
    //     );
    //   };

    //   const gradientStops = [
    //     { offset: 0, color: "#ff776b" },
    //     { offset: getOffset(6 + 2 * bandHalfSize), color: "#ffb36b" },
    //     { offset: getOffset(6 + bandHalfSize), color: "#FFCB6B" },
    //     { offset: getOffset(6), color: "#7fff6b" }, // target line
    //     { offset: getOffset(6 - bandHalfSize), color: "#FFCB6B" },
    //     { offset: getOffset(6 - 2 * bandHalfSize), color: "#ffb36b" },
    //     { offset: 100, color: "#ff776b" },
    //   ];

    //   gradientStops.forEach(({ offset, color }) => {
    //     rateGradient
    //       .append("stop")
    //       .attr("offset", `${offset}%`)
    //       .attr("stop-color", color);
    //   });

    //   g.append("rect")
    //     .attr("x", 0)
    //     .attr("y", 0)
    //     .attr("width", innerWidth)
    //     .attr("height", innerHeight)
    //     .attr("fill", "url(#rate-gradient)")
    //     .attr("opacity", 0.5)
    //     .lower();
    // }

    // const xAxisGroup = g
    //   .append("g")
    //   .attr("transform", `translate(0,${innerHeight})`)
    //   .call(
    //     hasXLabels
    //       ? d3
    //           .axisBottom(xScale)
    //           .tickValues(
    //             Array.from({ length: 10 }, (_, i) =>
    //               i === 9
    //                 ? xLabels[xLabels.length - 1]
    //                 : xLabels[Math.floor(i * ((xLabels.length - 1) / 9))]
    //             )
    //           )
    //       : d3.axisBottom(xScale).ticks(10)
    //   );

    const xAxisGroup = g
      .append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(0));

    const yAxisGroup = g.append("g").call(yAxis);

    // [xAxisGroup, yAxisGroup].forEach((axis) => {
    //   axis
    //     .selectAll("path, line")
    //     .attr("stroke", "#E0E0E0")
    //     .attr("stroke-opacity", 0.25);
    //   axis.selectAll("text").attr("fill", "#E0E0E0").attr("fill-opacity", 0.7);
    // });

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

    // g.append("text")
    //   .attr("x", innerWidth / 2)
    //   .attr("y", innerHeight + 40)
    //   .attr("text-anchor", "middle")
    //   .attr("fill", "#E0E0E0")
    //   .attr("opacity", 0.7)
    //   .text(xLabel);

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -35)
      .attr("x", -innerHeight / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#E0E0E0")
      .attr("opacity", 0.7)
      .text(yLabel);

    g.append("path")
      .datum(data)
      .attr("fill", "url(#line-gradient)")
      .attr("d", area);

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", lineColor)
      .attr("stroke-width", 2)
      .attr("d", line);

    highlightedSections.forEach(({ start, end }) => {
      g.append("rect")
        .attr("x", xScale(start))
        .attr("y", 0)
        .attr("width", xScale(end) - xScale(start))
        .attr("height", innerHeight)
        .attr("fill", "#FFCB6B")
        .attr("opacity", 0.01);
    });

    // Add an overlay to capture mouse events for hover interaction.
    const focus = g.append("g").style("display", "none");

    // Main circle on the line
    focus.append("circle").attr("r", 4.5).attr("fill", "#FF89BB");

    // Create a label group that appears above the main circle.
    // This group contains its own circle (the label bubble) and centered text.
    const labelGroup = focus
      .append("g")
      .attr("class", "labelGroup")
      .attr("transform", "translate(0, -30)"); // offset above the main circle

    // Append a rounded rectangle.
    // Adjust x, y, width, height, rx and ry as needed to fit your text.
    labelGroup
      .append("rect")
      .attr("x", -25) // center the rectangle relative to x=0
      .attr("y", -10) // vertical offset so text is centered
      .attr("width", 50) // adjust width based on expected text length
      .attr("height", 20)
      .attr("rx", 5) // rounded corners
      .attr("ry", 5)
      .attr("fill", "#FF89BB")
      .attr("opacity", 0.5); // semi-transparent background

    labelGroup
      .append("path")
      .attr("d", "M -5,10 L 0,15 L 5,10 Z")
      .attr("fill", "#FF89BB")
      .attr("opacity", 0.5);

    // Append text inside the rounded rectangle.
    labelGroup
      .append("text")
      .attr("x", 0)
      .attr("y", 0)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("fill", "#E0E0E0") // white text inside label rectangle
      .attr("opacity", 1) // slightly more opaque text
      .style("font-size", "10px");

    // Add an overlay rectangle to capture mouse events (over the entire chart area)
    g.append("rect")
      .attr("class", "overlay")
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .style("fill", "none")
      .style("pointer-events", "all")
      .on("mouseover", () => focus.style("display", null))
      .on("mouseout", () => focus.style("display", "none"))
      .on("mousemove", function (event) {
        const [mx, my] = d3.pointer(event);
        const x0 = xScale.invert(mx);
        const i = Math.round(x0);
        const dVal = data[i];

        if (dVal !== undefined) {
          const xCoord = xScale(i);
          const yCoord = yScale(dVal);

          // Check if the mouse y-position is close to the line's y-coordinate.
          const distance = Math.abs(my - yCoord);
          const threshold = 10; // pixels

          if (distance < threshold) {
            focus
              .style("display", null)
              .attr("transform", `translate(${xCoord},${yCoord})`);
            // Update the label text within the labelGroup.
            focus.select(".labelGroup").select("text").text(dVal.toFixed(2));
          } else {
            focus.style("display", "none");
          }
        } else {
          focus.style("display", "none");
        }
      });
  }, [
    data,
    width,
    height,
    highlightedSections,
    xLabel,
    yLabel,
    yMin,
    yMax,
    xLabels,
    feature,
    lineColor,
  ]);

  return <svg ref={ref} width={width} height={height} />;
};

export default LineGraph;
