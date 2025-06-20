import * as d3 from "d3";

export const createResetButton = (parentGroup, innerWidth, onReset) => {
  const resetButton = parentGroup
    .append("g")
    .attr("class", "reset-button")
    .attr("transform", `translate(${innerWidth - 60}, 10)`)
    .style("cursor", "pointer")
    .on("click", onReset)
    .on("mouseover", function () {
      // Change opacity to 100% on hover
      d3.select(this).select("rect").attr("opacity", 1.0);
    })
    .on("mouseout", function () {
      // Return to original opacity when not hovering
      d3.select(this).select("rect").attr("opacity", 0.6);
    });

  resetButton
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 50)
    .attr("height", 20)
    .attr("rx", 3)
    .attr("fill", "#1E1E2F")
    .attr("opacity", 0.6)
    .style("transition", "opacity 0.2s ease");

  resetButton
    .append("text")
    .attr("x", 25)
    .attr("y", 14)
    .attr("text-anchor", "middle")
    .attr("fill", "#E0E0E0")
    .attr("font-size", "10px")
    .text("reset");

  return resetButton;
};
