import { useRef, useEffect } from "react";
import * as d3 from "d3";

// Import utilities
import {
  processChartData,
  createXScale,
  createZoomedXScale,
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
  createHighlightedSections,
  updateHighlightedSections,
  createMainChart,
  updateMainChart,
  createSilenceIndicators,
  updateSilenceIndicators,
} from "./components";

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
  onZoomChange,
}) => {
  const ref = useRef();
  const lastChangeRef = useRef(null);

  useEffect(() => {
    // Early return if no data
    const chartData = processChartData(data, yMin, yMax, height - 40);
    if (!chartData.isValid) {
      console.error(chartData.error);
      return;
    }

    const { filteredData, yDomain, yScale } = chartData;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 20, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create scales
    const xScale = createXScale(data, xLabels, innerWidth);

    // Create container group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create definitions and clip path
    const defs = svg.append("defs");
    createClipPath(defs, "chart-clip", innerWidth, innerHeight);

    // Create gradient for non-rates/extents features
    if (feature !== "rates" && feature !== "extents") {
      const gradientStops = getDefaultLineGradientStops();
      createGradient(defs, "line-gradient", gradientStops);
    }

    // Keep track of current scale for zooming
    let currentXScale = xScale;

    // Create chart group with clip path
    const chartGroup = g.append("g").attr("clip-path", "url(#chart-clip)");

    // Create background elements
    if (feature === "pitch" || feature === "vibrato") {
      createNoteStripes(g, yScale, yDomain, innerWidth);
    }
    createFeatureBackground(g, defs, feature, yDomain, innerWidth, innerHeight);

    // Create axes
    createAxes(g, xScale, yScale, yDomain, feature, innerHeight, yLabel);

    // Create highlighted sections
    createHighlightedSections(
      chartGroup,
      highlightedSections,
      xScale,
      innerHeight
    );

    // Create main chart (line and area)
    createMainChart(
      chartGroup,
      filteredData,
      xScale,
      yScale,
      yDomain,
      lineColor
    );

    // Create silence indicators
    createSilenceIndicators(chartGroup, filteredData, xScale, innerHeight);

    // Create tooltip
    const focus = createTooltip(g);

    const calculateZoomCoordinates = (scale) => {
      const domain = scale.domain();
      return {
        startIndex: Math.max(0, Math.round(domain[0])),
        endIndex: Math.min(data.length - 1, Math.round(domain[1])),
        startX: scale(domain[0]),
        endX: scale(domain[1]),
        zoomLevel: domain, // [start, end] in data coordinates
        isZoomed: scale !== xScale, // Check if this is not the original scale
        visibleDataPercentage:
          ((domain[1] - domain[0]) / (data.length - 1)) * 100,
      };
    };

    // Function to safely notify about highlight changes (only when they actually change)
    const notifyChange = (newXScale) => {
      const zoomCoordinates = calculateZoomCoordinates(newXScale);

      const changeData = {
        zoom: zoomCoordinates,
      };

      const changeString = JSON.stringify(changeData);

      if (lastChangeRef.current !== changeString) {
        lastChangeRef.current = changeString;
        if (onZoomChange) {
          onZoomChange(changeData);
        }
      }
    };

    // Function to redraw chart elements
    function redrawChart(newXScale, currentYScale) {
      currentXScale = newXScale;

      // Update main chart
      updateMainChart(
        chartGroup,
        filteredData,
        newXScale,
        currentYScale,
        yDomain
      );

      // Update highlighted sections
      if (highlightedSections.length > 0) {
        updateHighlightedSections(chartGroup, newXScale);
      }

      // Update silence indicators
      updateSilenceIndicators(chartGroup, filteredData, newXScale, innerHeight);

      // Calculate and notify about updated highlight coordinates (only if changed)
      notifyChange(newXScale);
    }

    // Create brush for zoom selection
    const brush = d3
      .brushX()
      .extent([
        [0, 0],
        [innerWidth, innerHeight],
      ])
      .on("end", function (event) {
        if (!event.selection) return;

        const [x0, x1] = event.selection;
        const newXScale = createZoomedXScale(currentXScale, [x0, x1]);

        // Clear the brush selection
        d3.select(this).call(brush.clear);

        // Redraw with new scale
        redrawChart(newXScale, yScale);

        // Update the current scale for future operations
        currentXScale = newXScale;
      });

    // Add brush overlay
    const brushOverlay = chartGroup
      .append("g")
      .attr("class", "brush")
      .call(brush);

    // Style the brush
    brushOverlay
      .selectAll(".overlay")
      .style("cursor", "crosshair")
      .style("pointer-events", "all");

    brushOverlay
      .selectAll(".selection")
      .style("fill", "#FF89BB")
      .style("fill-opacity", 0.2)
      .style("stroke", "#FF89BB")
      .style("stroke-width", 1);

    // Add mouse events for tooltips
    brushOverlay
      .on("mouseover", () => {
        focus.style("display", null);
      })
      .on("mouseout", () => {
        hideTooltip(focus);
      })
      .on("mousemove", function (event) {
        const [mx, my] = d3.pointer(event);
        const x0 = currentXScale.invert(mx);
        const i = Math.round(x0);
        const dVal = data[i];
        const filteredVal = filteredData[i];

        if (dVal !== undefined && i >= 0 && i < data.length) {
          const xCoord = currentXScale(i);
          const isSilence = filteredVal === null;

          const yCoord = isSilence ? innerHeight - 5 : yScale(filteredVal);
          const threshold = isSilence ? 20 : 10;
          const distance = Math.abs(my - (isSilence ? innerHeight : yCoord));

          if (distance < threshold || (isSilence && my > innerHeight - 30)) {
            const displayText = isSilence ? "Silence" : dVal.toFixed(2);
            updateTooltip(focus, xCoord, yCoord, displayText, isSilence);
          } else {
            hideTooltip(focus);
          }
        } else {
          hideTooltip(focus);
        }
      });

    // Create reset button
    const handleReset = () => {
      currentXScale = xScale;
      redrawChart(xScale, yScale);
    };

    createResetButton(g, innerWidth, handleReset);

    // Initial call to set the highlighted sections coordinates (only if changed)
    // notifyChange(xScale);
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
    onZoomChange,
  ]);

  return <svg ref={ref} width={width} height={height} />;
};

export default LineGraph;
