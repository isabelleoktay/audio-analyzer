import { generateNoteStripeData, getFeatureColorStops } from "../utils";
import { createVerticalBackgroundGradient } from "../../../../utils/createVerticalBackgroundGradient";

export const createNoteStripes = (parentGroup, yScale, yDomain, innerWidth) => {
  const stripeData = generateNoteStripeData(yDomain);

  stripeData.forEach((stripe) => {
    const yTop = yScale(stripe.end);
    const yBottom = yScale(stripe.start);
    const stripeHeight = yBottom - yTop;

    parentGroup
      .append("rect")
      .attr("x", 0)
      .attr("y", yTop)
      .attr("width", innerWidth)
      .attr("height", stripeHeight)
      .attr("fill", stripe.color)
      .attr("opacity", stripe.opacity)
      .lower();
  });
};

export const createFeatureBackground = (
  parentGroup,
  defs,
  feature,
  yDomain,
  innerWidth,
  innerHeight
) => {
  if (feature === "extents" || feature === "rates") {
    const colorStops = getFeatureColorStops(feature);

    const gradientFill = createVerticalBackgroundGradient({
      svgDefs: defs,
      id: "extent-gradient",
      yDomain,
      colorStops,
    });

    parentGroup
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("fill", gradientFill)
      .attr("opacity", 0.25)
      .lower();
  }
};
