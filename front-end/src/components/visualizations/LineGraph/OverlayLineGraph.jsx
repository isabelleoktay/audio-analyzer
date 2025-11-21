import { useRef, useEffect, useMemo } from "react";
import * as d3 from "d3";

import {
  processChartData,
  createXScale,
  createZoomedXScale,
  createGradient,
  getDefaultLineGradientStops,
  createClipPath,
} from "./utils";

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
  highlightedSections = [],
  yMin,
  yMax,
  xLabels,
  zoomDomain,
  onZoomChange,
  onSimilarityCalculated,
}) => {
  const ref = useRef();
  const lastChangeRef = useRef(null);

  const margin = useMemo(
    () => ({ top: 20, right: 20, bottom: 20, left: 50 }),
    []
  );
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // X scale (applies zoomDomain if provided)
  const xScale = useMemo(() => {
    const scale = createXScale(primaryData, xLabels, innerWidth);
    if (zoomDomain) scale.domain(zoomDomain);
    return scale;
  }, [primaryData, xLabels, innerWidth, zoomDomain]);

  useEffect(() => {
    if (!primaryData || primaryData.length === 0) return;

    const chartData = processChartData(
      primaryData,
      Array.isArray(secondaryData) ? secondaryData : [],
      yMin,
      yMax,
      innerHeight
    );

    if (!chartData.isValid) {
      console.warn("OverlayLineGraph: No valid data", chartData.error);
      return;
    }

    const { filteredPrimaryData, filteredSecondaryData, yDomain, yScale } =
      chartData;

    if (filteredSecondaryData && filteredSecondaryData.length > 0) {
      const similarityScore = similarityFromArea(
        filteredPrimaryData,
        filteredSecondaryData
      );
      // Send similarity to parent
      if (typeof onSimilarityCalculated === "function") {
        onSimilarityCalculated(similarityScore);
      }
    }
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const defs = svg.append("defs");
    createClipPath(defs, "chart-clip", innerWidth, innerHeight);

    if (feature !== "rates" && feature !== "extents") {
      createGradient(defs, "line-gradient", getDefaultLineGradientStops());
    }

    createGradient(
      defs,
      "secondary-line-gradient",
      getDefaultLineGradientStops(secondaryLineColor)
    );

    let currentXScale = xScale;

    const chartGroup = g.append("g").attr("clip-path", "url(#chart-clip)");

    if (feature === "pitch") createNoteStripes(g, yScale, yDomain, innerWidth);
    createFeatureBackground(g, defs, feature, yDomain, innerWidth, innerHeight);

    // eslint-disable-next-line no-unused-vars
    const { xAxisGroup, yAxisGroup } = createAxes(
      g,
      xScale,
      yScale,
      yDomain,
      feature,
      innerHeight,
      yLabel
    );

    createHighlightedSections(
      chartGroup,
      highlightedSections,
      xScale,
      innerHeight,
      feature
    );

    createMainChart(
      chartGroup,
      filteredPrimaryData,
      primaryLineColor,
      filteredSecondaryData,
      secondaryLineColor,
      xScale,
      yScale,
      yDomain
    );

    createSilenceIndicators(
      chartGroup,
      filteredPrimaryData,
      xScale,
      innerHeight
    );

    const focus = createTooltip(g);

    const calculateZoomCoordinates = (scale) => {
      const domain = scale.domain();
      return {
        startIndex: Math.max(0, Math.round(domain[0])),
        endIndex: Math.min(primaryData.length - 1, Math.round(domain[1])),
        startX: scale(domain[0]),
        endX: scale(domain[1]),
        zoomLevel: domain,
        isZoomed: scale !== xScale,
        visibleDataPercentage:
          ((domain[1] - domain[0]) / (primaryData.length - 1)) * 100,
      };
    };

    const notifyChange = (newXScale) => {
      const zoomCoordinates = calculateZoomCoordinates(newXScale);
      const changeData = { zoom: zoomCoordinates };
      const changeString = JSON.stringify(changeData);

      if (lastChangeRef.current !== changeString) {
        lastChangeRef.current = changeString;
        onZoomChange?.(changeData);
      }
    };

    function similarityFromArea(curveA, curveB) {
      if (!curveA || !curveB || curveA.length === 0 || curveB.length === 0)
        return null;

      const n = Math.min(curveA.length, curveB.length);
      if (n < 2) return null;

      const isYOnly = typeof curveA[0] === "number";

      // Compute yMin and yMax across both curves
      let yMin = Infinity;
      let yMax = -Infinity;
      for (let i = 0; i < n; i++) {
        const yA = isYOnly ? curveA[i] : curveA[i][1];
        const yB = isYOnly ? curveB[i] : curveB[i][1];
        if (yA != null) {
          yMin = Math.min(yMin, yA);
          yMax = Math.max(yMax, yA);
        }
        if (yB != null) {
          yMin = Math.min(yMin, yB);
          yMax = Math.max(yMax, yB);
        }
      }

      if (yMax === yMin) return 100; // curves are identical flat line

      // Calculate area between curves
      let totalArea = 0;
      for (let i = 1; i < n; i++) {
        const x0 = isYOnly ? i - 1 : curveA[i - 1][0];
        const x1 = isYOnly ? i : curveA[i][0];

        const yA0 = isYOnly ? curveA[i - 1] : curveA[i - 1][1];
        const yA1 = isYOnly ? curveA[i] : curveA[i][1];

        const yB0 = isYOnly ? curveB[i - 1] : curveB[i - 1][1];
        const yB1 = isYOnly ? curveB[i] : curveB[i][1];

        const polyX = [x0, x1, x1, x0];
        const polyY = [yA0, yA1, yB1, yB0];

        // Shoelace formula for quadrilateral area
        let quadArea = 0;
        for (let j = 0; j < 4; j++) {
          const jNext = (j + 1) % 4;
          quadArea += polyX[j] * polyY[jNext] - polyX[jNext] * polyY[j];
        }
        totalArea += Math.abs(quadArea / 2);
      }

      // Maximum possible area between two curves
      const maxArea = (yMax - yMin) * (n - 1);
      const similarity = Math.max(0, 100 * (1 - totalArea / maxArea));

      return similarity;
    }

    function redrawChart(newXScale, currentYScale) {
      currentXScale = newXScale;

      const [startIndex, endIndex] = newXScale
        .domain()
        .map((d) =>
          Math.max(0, Math.min(filteredPrimaryData.length - 1, Math.round(d)))
        );

      const zoomedPrimaryData = filteredPrimaryData.slice(
        startIndex,
        endIndex + 1
      );
      const zoomedSecondaryData = filteredSecondaryData.length
        ? filteredSecondaryData.slice(startIndex, endIndex + 1)
        : [];

      updateMainChart(
        chartGroup,
        zoomedPrimaryData,
        zoomedSecondaryData,
        newXScale,
        currentYScale,
        yDomain
      );

      xAxisGroup.call(d3.axisBottom(newXScale));

      if (highlightedSections.length > 0) {
        updateHighlightedSections(chartGroup, newXScale);
      }

      updateSilenceIndicators(
        chartGroup,
        filteredPrimaryData,
        newXScale,
        innerHeight
      );
      notifyChange(newXScale);
    }

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

        d3.select(this).call(brush.clear);
        redrawChart(newXScale, yScale);
        currentXScale = newXScale;
      });

    const brushOverlay = chartGroup
      .append("g")
      .attr("class", "brush")
      .call(brush);

    brushOverlay
      .selectAll(".overlay")
      .style("cursor", "crosshair")
      .style("pointer-events", "all");

    brushOverlay
      .selectAll(".selection")
      .style("fill", "#90F1EF")
      .style("fill-opacity", 0.2)
      .style("stroke", "#90F1EF")
      .style("stroke-width", 1);

    brushOverlay
      .on("mouseover", () => focus.style("display", null))
      .on("mouseout", () => hideTooltip(focus))
      .on("mousemove", function (event) {
        const [mx, my] = d3.pointer(event);
        const x0 = currentXScale.invert(mx);
        const i = Math.round(x0);
        const dVal = primaryData[i];
        const filteredVal = filteredPrimaryData[i];

        if (dVal !== undefined && i >= 0 && i < primaryData.length) {
          const xCoord = currentXScale(i);
          const isSilence = filteredVal === null;
          const yCoord = isSilence ? innerHeight - 5 : yScale(filteredVal);
          const threshold = isSilence ? 20 : 10;
          const distance = Math.abs(my - (isSilence ? innerHeight : yCoord));

          if (distance < threshold || (isSilence && my > innerHeight - 30)) {
            const displayText = isSilence ? "Silence" : dVal.toFixed(2);
            updateTooltip(
              focus,
              xCoord,
              yCoord,
              displayText,
              isSilence,
              innerHeight
            );
          } else hideTooltip(focus);
        } else hideTooltip(focus);
      });

    const handleReset = () => {
      // Reset the scale domain to the full range
      const resetXScale = createXScale(primaryData, xLabels, innerWidth);
      currentXScale = resetXScale;

      // Redraw everything with the reset scale
      redrawChart(resetXScale, yScale);

      // Notify the parent that zoom is cleared
      notifyChange(resetXScale);
    };

    createResetButton(g, innerWidth, handleReset);
  }, [
    primaryData,
    secondaryData,
    width,
    height,
    innerWidth,
    innerHeight,
    xScale,
    margin,
    highlightedSections,
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

  return <svg ref={ref} width={width} height={height} />;
};

export default OverlayLineGraph;
