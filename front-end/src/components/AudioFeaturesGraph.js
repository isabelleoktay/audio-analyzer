import React, { useRef, useEffect, useState } from "react";

const AudioFeaturesGraph = ({
  data,
  xLabels,
  yLabels,
  minY,
  maxY,
  color = "black",
  highlightedSections = [],
}) => {
  const baseCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [resizedWindow, setResizedWindow] = useState(false);

  // Base layer setup
  useEffect(() => {
    const resizeCanvases = () => {
      [baseCanvasRef, overlayCanvasRef].forEach((ref) => {
        const canvas = ref.current;
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth * dpr;
        const height = canvas.clientHeight * dpr;
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d").scale(dpr, dpr);
      });
    };

    const drawAxes = (context, width, height, padding) => {
      context.strokeStyle = "black";
      context.lineWidth = 1;

      context.font = "12px 'Poppins', sans-serif";

      const getTextWidth = (text) => context.measureText(text).width;
      const maxLabelWidth = Math.max(
        ...yLabels.map((label) => getTextWidth(label.label))
      );

      const yLabelMargin = maxLabelWidth - 12;

      context.beginPath();
      context.moveTo(padding, height - padding);
      context.lineTo(width - padding, height - padding);
      context.stroke();

      context.beginPath();
      context.moveTo(padding, height - padding);
      context.lineTo(padding, padding);
      context.stroke();

      context.font = "12px 'Poppins', sans-serif";
      xLabels.forEach((label) => {
        const x = padding + label.position * (width - 2 * padding);
        context.fillText(label.label, x, height - padding / 2);
      });

      context.font = "12px 'Poppins', sans-serif";
      yLabels.forEach((label) => {
        const y = height - padding - label.position * (height - 2 * padding);
        context.fillText(label.label, padding / 2 - yLabelMargin, y);
      });
    };

    const drawData = (context, width, height, padding) => {
      const drawValues = (data, color) => {
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.beginPath();
        data.forEach((value, index) => {
          const x =
            padding + (index / (data.length - 1)) * (width - 2 * padding);
          const y =
            height -
            padding -
            ((value - minY) / (maxY - minY)) * (height - 2 * padding);
          if (index === 0) {
            context.moveTo(x, y);
          } else {
            context.lineTo(x, y);
          }
        });
        context.stroke();
      };

      if (typeof data[0] === "object") {
        data.forEach((line) => drawValues(line.data, line.lineColor));
      } else {
        drawValues(data, color);
      }
    };

    const drawBase = () => {
      const canvas = baseCanvasRef.current;
      const context = canvas.getContext("2d");
      const padding = 40;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      context.clearRect(0, 0, width, height);
      drawAxes(context, width, height, padding);
      drawData(context, width, height, padding);
    };

    resizeCanvases();
    drawBase();

    const handleResize = () => {
      resizeCanvases();
      setResizedWindow(true);
      drawBase();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [data, xLabels, yLabels, minY, maxY, color]);

  // Highlight updates
  useEffect(() => {
    const drawHighlights = (context, width, height, padding) => {
      highlightedSections.forEach((section) => {
        const { start, end, color: sectionColor, label } = section;
        const startX =
          padding + (start / (data.length - 1)) * (width - 2 * padding);
        const endX =
          padding + (end / (data.length - 1)) * (width - 2 * padding);
        const sectionWidth = endX - startX;

        context.fillStyle = sectionColor;
        context.globalAlpha = 0.3;
        context.fillRect(startX, padding, sectionWidth, height - 2 * padding);
        context.globalAlpha = 1;

        context.fillStyle = color;
        context.font = "12px 'Poppins', sans-serif";
        context.textAlign = "center";

        const labelY = padding - 10;
        context.fillText(label, startX + sectionWidth / 2, labelY);
      });
    };

    const updateHighlights = () => {
      const canvas = overlayCanvasRef.current;
      const context = canvas.getContext("2d");
      const padding = 40;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      context.clearRect(padding, 0, width - padding, height - padding);
      drawHighlights(context, width, height, padding);
      setResizedWindow(false);
    };

    if (overlayCanvasRef.current) {
      updateHighlights();
    }
  }, [highlightedSections, data, color, resizedWindow]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={baseCanvasRef}
        className="absolute top-0 left-0 w-full h-full"
      />
      <canvas
        ref={overlayCanvasRef}
        className="absolute top-0 left-0 w-full h-full"
      />
    </div>
  );
};

export default AudioFeaturesGraph;
