import { getTooltipColors } from "../utils";

export const createTooltip = (parentGroup) => {
  const focus = parentGroup.append("g").style("display", "none");

  // Main circle on the line
  focus.append("circle").attr("r", 4.5).attr("fill", "#FF89BB");

  // Create a label group that appears above the main circle.
  const labelGroup = focus
    .append("g")
    .attr("class", "labelGroup")
    .attr("transform", "translate(0, -30)");

  // Append a rounded rectangle.
  labelGroup
    .append("rect")
    .attr("x", -25)
    .attr("y", -10)
    .attr("width", 50)
    .attr("height", 20)
    .attr("rx", 5)
    .attr("ry", 5)
    .attr("fill", "#FF89BB")
    .attr("opacity", 0.5);

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
    .attr("fill", "#E0E0E0")
    .attr("opacity", 1)
    .style("font-size", "10px");

  return focus;
};

export const updateTooltip = (
  focus,
  xCoord,
  yCoord,
  displayText,
  isSilence
) => {
  focus
    .style("display", null)
    .attr("transform", `translate(${xCoord},${yCoord})`);

  const labelGroup = focus.select(".labelGroup");
  const rect = labelGroup.select("rect");

  const bbox = labelGroup.node().getBBox();
  const tooltipHeight = bbox.height + 10;

  // Position above by default
  let offsetY = -tooltipHeight;

  const wouldOverflowTop = yCoord - tooltipHeight < 0;
  if (wouldOverflowTop) {
    offsetY = 15; // place below point
    labelGroup.select("path").attr("d", "M -5,-10 L 0,-15 L 5,-10 Z");
  } else {
    labelGroup.select("path").attr("d", "M -5,10 L 0,15 L 5,10 Z");
  }

  // Apply Y offset
  labelGroup.attr("transform", `translate(0, ${offsetY})`);

  labelGroup.select("text").text(displayText);
  const { bgColor, textColor, opacity } = getTooltipColors(isSilence);

  focus
    .select("circle")
    .attr("fill", isSilence ? "#555555" : "#FF89BB")
    .attr("opacity", isSilence ? 0.8 : 1);

  rect.attr("fill", bgColor).attr("opacity", opacity);
  labelGroup.select("path").attr("fill", bgColor).attr("opacity", opacity);
  labelGroup.select("text").attr("fill", textColor);
};

export const hideTooltip = (focus) => {
  focus.style("display", "none");
};
