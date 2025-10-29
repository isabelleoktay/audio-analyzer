import { useRef, useEffect } from "react";
import * as d3 from "d3";

// Import utilities
import {
  processChartDataNew,
  createXScale,
  createGradient,
  getDefaultLineGradientStops,
  createClipPath,
} from "./utils";

// Import components
import {
  createTooltip,
  updateTooltip,
  hideTooltip,
  createResetButton,
  createAxes,
  createNoteStripes,
  createFeatureBackground,
  createMainChartNew,
  updateMainChartNew,
} from "./components";

const OverlayLineGraph = ({
  feature = "pitch",
  primaryData = [],
  primaryLineColor = "#FF89BB",
  secondaryData = [],
  secondaryLineColor = "#CCCCCC",
  width = 800,
  height = 400,
  xLabel,
  yLabel,
  yMin,
  yMax,
  xLabels,
  onZoomChange,
}) => {
  const svgRef = useRef();
  const lastChangeRef = useRef(null);

  useEffect(() => {
    // Safety: don't render if no primary data
    if (!primaryData || primaryData.length === 0) return;

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .style("background", "transparent");

    svg.selectAll("*").remove(); // clear previous render

    // Process chart data safely
    const chartData = processChartDataNew(
      primaryData,
      Array.isArray(secondaryData) ? secondaryData : [],
      yMin,
      yMax,
      height - 40
    );

    if (!chartData.isValid) {
      console.warn("OverlayLineGraph: No valid data", chartData.error);
      return;
    } 

    const margin = { top: 20, right: 20, bottom: 20, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const { filteredPrimaryData, filteredSecondaryData, yDomain, yScale } =
      chartData;

    // Create scales
    const xScale = createXScale(primaryData, xLabels, innerWidth);

    // Container group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Definitions and clip path
    const defs = svg.append("defs");
    createClipPath(defs, "chart-clip", innerWidth, innerHeight);

    if (feature !== "rates" && feature !== "extents") {
      const gradientStops = getDefaultLineGradientStops();
      createGradient(defs, "line-gradient", gradientStops);
    }

    const chartGroup = g.append("g").attr("clip-path", "url(#chart-clip)");

    // Background elements
    if (feature === "pitch" || feature === "vibrato") {
      createNoteStripes(g, yScale, yDomain, innerWidth);
    }
    createFeatureBackground(g, defs, feature, yDomain, innerWidth, innerHeight);

    // Axes
    createAxes(g, xScale, yScale, yDomain, feature, innerHeight, yLabel);

    // Main chart (primary + optional secondary)
    createMainChartNew(
      chartGroup,
      filteredPrimaryData,
      primaryLineColor,
      filteredSecondaryData,
      secondaryLineColor,
      xScale,
      yScale,
      yDomain
    );

    // Tooltip
    const focus = createTooltip(g);

    // Redraw function (for zoom or updates)
    const redrawChart = (newXScale) => {
      updateMainChartNew(
        chartGroup,
        filteredPrimaryData,
        filteredSecondaryData,
        newXScale,
        yScale,
        yDomain
      );
    };

    // Brush (zoom) functionality
    const brush = d3
      .brushX()
      .extent([
        [0, 0],
        [innerWidth, innerHeight],
      ])
      .on("end", function (event) {
        if (!event.selection) return;
        const [x0, x1] = event.selection;

        const newXScale = xScale
          .copy()
          .domain([xScale.invert(x0), xScale.invert(x1)]);

        d3.select(this).call(brush.clear);
        redrawChart(newXScale);

        // Notify parent about zoom
        if (onZoomChange) {
          const startIndex = Math.max(0, Math.round(xScale.invert(x0)));
          const endIndex = Math.min(
            primaryData.length - 1,
            Math.round(xScale.invert(x1))
          );
          const zoomData = {
            startIndex,
            endIndex,
            isZoomed: true,
            zoomLevel: [xScale.invert(x0), xScale.invert(x1)],
          };
          const changeString = JSON.stringify(zoomData);
          if (lastChangeRef.current !== changeString) {
            lastChangeRef.current = changeString;
            onZoomChange({ zoom: zoomData });
          }
        }
      });

    chartGroup.append("g").attr("class", "brush").call(brush);

    // Reset button
    createResetButton(g, innerWidth, () => redrawChart(xScale));
  }, [
    primaryData,
    secondaryData,
    width,
    height,
    xLabel,
    yLabel,
    yMin,
    yMax,
    xLabels,
    feature,
    primaryLineColor,
    secondaryLineColor,
    onZoomChange,
  ]);

  return <svg ref={svgRef} width={width} height={height} />;
};

export default OverlayLineGraph;
