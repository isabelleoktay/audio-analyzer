import React, { useRef, useEffect, useState } from "react";

const AudioFeaturesGraph = ({
  data,
  xLabels,
  yLabels,
  minY,
  maxY,
  color = "black",
  highlightedSections = [],
  onRenderComplete, // Callback to signal rendering completion
}) => {
  const baseCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [resizedWindow, setResizedWindow] = useState(false);
  const [renderCount, setRenderCount] = useState(0); // Track rendering completionxw

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
      const drawValues = (data, color, dashed = false) => {
        context.strokeStyle = color;
        context.setLineDash(dashed ? [5, 10] : []);
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
        data.forEach((line) =>
          drawValues(line.data, line.lineColor, line.dashed)
        );
      } else {
        drawValues(data, color);
      }
    };

    const drawBase = () => {
      const canvas = baseCanvasRef.current;
      const context = canvas.getContext("2d");
      const padding = 50;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      context.clearRect(0, 0, width, height);
      drawAxes(context, width, height, padding);
      drawData(context, width, height, padding);

      // Increment render count
      setRenderCount((prev) => prev + 1);
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

        // Normalize start and end relative to the data length
        const normalizedStart = start / (data.length - 1);
        const normalizedEnd = end / (data.length - 1);

        // Calculate startX and endX, taking padding into account
        const startX = padding + normalizedStart * (width - 2 * padding);
        const endX = padding + normalizedEnd * (width - 2 * padding);
        const sectionWidth = endX - startX;

        // Draw the highlighted section
        context.fillStyle = sectionColor;
        context.globalAlpha = 0.3;
        context.fillRect(startX, padding, sectionWidth, height - 2 * padding);
        context.globalAlpha = 1;

        // Draw the label
        context.fillStyle = color;
        context.font = "12px 'Poppins', sans-serif";
        context.textAlign = "center";

        const labelY = padding - 10;
        const maxWidth = sectionWidth - 4; // Padding inside the highlight

        // Function to wrap text within maxWidth
        const wrapText = (text, maxWidth) => {
          const words = text.split(" ");
          let lines = [];
          let currentLine = words[0];

          for (let i = 1; i < words.length; i++) {
            const testLine = currentLine + " " + words[i];
            if (context.measureText(testLine).width < maxWidth) {
              currentLine = testLine;
            } else {
              lines.push(currentLine);
              currentLine = words[i];
            }
          }
          lines.push(currentLine);
          return lines;
        };

        const wrappedLines = wrapText(label, maxWidth);
        const lineHeight = 14; // Line spacing

        // Draw wrapped lines centered inside highlight section
        wrappedLines.forEach((line, index) => {
          context.fillText(
            line,
            startX + sectionWidth / 2,
            labelY - index * lineHeight
          );
        });
      });
    };

    const updateHighlights = () => {
      const canvas = overlayCanvasRef.current;
      const context = canvas.getContext("2d");
      const padding = 50;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      context.clearRect(0, 0, width, height);
      drawHighlights(context, width, height, padding);
      setResizedWindow(false);

      // Increment render count
      setRenderCount((prev) => prev + 1);
    };

    if (overlayCanvasRef.current) {
      updateHighlights();
    }
  }, [highlightedSections, data, color, resizedWindow]);

  // Signal parent when both canvases are rendered
  useEffect(() => {
    if (onRenderComplete) {
      requestAnimationFrame(() => {
        onRenderComplete();
      });
    }
  }, [renderCount, onRenderComplete]);

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
