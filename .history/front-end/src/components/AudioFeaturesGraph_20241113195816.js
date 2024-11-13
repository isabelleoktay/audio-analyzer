import React, { useRef, useEffect } from "react";

const AudioFeaturesGraph = ({
  datasets = [], // Array of objects, each { data, color, label }
  xLabels,
  yLabels,
  minY,
  maxY,
  highlightedSections = [],
  color = "black",
}) => {
  const baseCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const lineCanvasRefs = useRef([]); // Multiple refs for each dataset layer

  console.log(datasets);

  const resizeCanvases = () => {
    [baseCanvasRef, overlayCanvasRef, ...lineCanvasRefs.current].forEach(
      (ref) => {
        const canvas = ref.current;
        if (canvas) {
          const dpr = window.devicePixelRatio || 1;
          const width = canvas.clientWidth * dpr;
          const height = canvas.clientHeight * dpr;
          canvas.width = width;
          canvas.height = height;
          canvas.getContext("2d").scale(dpr, dpr);
        }
      }
    );
  };

  const drawAxes = (context, width, height, padding) => {
    context.strokeStyle = "black";
    context.lineWidth = 1;

    const getTextWidth = (text) => {
      return context.measureText(text).width;
    };

    const maxLabelWidth = Math.max(
      ...yLabels.map((label) => getTextWidth(label.label))
    );

    // Add some padding to the maximum width
    const yLabelMargin = maxLabelWidth - 12; // 10px extra padding

    // Draw X-Axis
    context.beginPath();
    context.moveTo(padding, height - padding);
    context.lineTo(width - padding, height - padding);
    context.stroke();

    // Draw Y-Axis
    context.beginPath();
    context.moveTo(padding, height - padding);
    context.lineTo(padding, padding);
    context.stroke();

    // Add X-axis labels
    xLabels.forEach((label) => {
      const x = padding + label.position * (width - 2 * padding);
      context.fillText(label.label, x, height - padding / 2);
    });

    // Draw Y-Axis labels (Amplitude/Frequency)
    yLabels.forEach((label) => {
      const y = height - padding - label.position * (height - 2 * padding);
      context.fillText(label.label, padding / 2 - yLabelMargin, y);
    });
  };

  const drawData = (context, data, width, height, padding, color) => {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    data.forEach((value, index) => {
      const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
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

  const drawHighlights = (context, width, height, padding) => {
    highlightedSections.forEach((section) => {
      const { start, end, color: sectionColor, label } = section;
      const audioData = datasets[0].data;
      const startX =
        padding + (start / audioData.length) * (width - 3 * padding);
      const endX = padding + (end / audioData.length) * (width - 3 * padding);
      const sectionWidth = endX - startX;

      context.fillStyle = sectionColor;
      context.globalAlpha = 0.3; // Make the highlight semi-transparent
      context.fillRect(startX, padding, sectionWidth, height - 2 * padding);
      context.globalAlpha = 1; // Reset alpha to default

      // Draw section labels just above the highlight
      context.fillStyle = color;
      context.font = "12px Arial"; // Set font style for the label
      context.textAlign = "center";

      // Position label just above the highlighted section
      const labelY = padding - 10; // 10 pixels above the top padding
      context.fillText(label, startX + sectionWidth / 2, labelY);
    });
  };

  // Highlights only on overlay
  const updateHighlights = () => {
    const canvas = overlayCanvasRef.current;
    const context = canvas.getContext("2d");
    const padding = 40;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Only clear the graph area, preserving axes
    context.clearRect(padding, 0, width - padding, height - padding);
    drawHighlights(context, width, height, padding);
  };

  const drawBase = () => {
    const canvas = baseCanvasRef.current;
    const context = canvas.getContext("2d");
    const padding = 40;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    context.clearRect(0, 0, width, height);
    drawAxes(context, width, height, padding);
  };

  const drawLineData = () => {
    const padding = 40;
    const width = baseCanvasRef.current.clientWidth;
    const height = baseCanvasRef.current.clientHeight;

    console.log("DRAWING LINE DATA");

    datasets.forEach((dataset, idx) => {
      const canvas = lineCanvasRefs.current[idx].current;
      if (canvas) {
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, width, height);
        console.log("drawing data:" + dataset.color);
        drawData(context, dataset.data, width, height, padding, dataset.color);
      }
    });
  };

  useEffect(() => {
    resizeCanvases();
    drawBase();
    drawLineData();

    const handleResize = () => {
      resizeCanvases();
      drawBase();
      drawLineData();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [datasets]);

  useEffect(() => {
    if (overlayCanvasRef.current) {
      updateHighlights();
    }
  }, [highlightedSections]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={baseCanvasRef}
        className="absolute top-0 left-0 w-full h-full"
      />
      {datasets.map((_, idx) => (
        <canvas
          key={idx}
          ref={(el) => (lineCanvasRefs.current[idx] = el)}
          className="absolute top-0 left-0 w-full h-full"
        />
      ))}
      <canvas
        ref={overlayCanvasRef}
        className="absolute top-0 left-0 w-full h-full"
      />
    </div>
  );
};

export default AudioFeaturesGraph;

// import React, { useRef, useEffect } from "react";

// const AudioFeaturesGraph = ({
//   data,
//   xLabels,
//   yLabels,
//   minY,
//   maxY,
//   color = "black",
//   highlightedSections = [],
// }) => {
//   const baseCanvasRef = useRef(null);
//   const overlayCanvasRef = useRef(null);

//   const resizeCanvases = () => {
//     [baseCanvasRef, overlayCanvasRef].forEach((ref) => {
//       const canvas = ref.current;
//       const dpr = window.devicePixelRatio || 1;
//       const width = canvas.clientWidth * dpr;
//       const height = canvas.clientHeight * dpr;
//       canvas.width = width;
//       canvas.height = height;
//       canvas.getContext("2d").scale(dpr, dpr);
//     });
//   };

//   const drawAxes = (context, width, height, padding) => {
//     context.strokeStyle = "black";
//     context.lineWidth = 1;

//     const getTextWidth = (text) => {
//       return context.measureText(text).width;
//     };

//     const maxLabelWidth = Math.max(
//       ...yLabels.map((label) => getTextWidth(label.label))
//     );

//     // Add some padding to the maximum width
//     const yLabelMargin = maxLabelWidth - 12; // 10px extra padding

//     // Draw X-Axis
//     context.beginPath();
//     context.moveTo(padding, height - padding);
//     context.lineTo(width - padding, height - padding);
//     context.stroke();

//     // Draw Y-Axis
//     context.beginPath();
//     context.moveTo(padding, height - padding);
//     context.lineTo(padding, padding);
//     context.stroke();

//     // Add X-axis labels
//     xLabels.forEach((label) => {
//       const x = padding + label.position * (width - 2 * padding);
//       context.fillText(label.label, x, height - padding / 2);
//     });

//     // Draw Y-Axis labels (Amplitude/Frequency)
//     yLabels.forEach((label) => {
//       const y = height - padding - label.position * (height - 2 * padding);
//       context.fillText(label.label, padding / 2 - yLabelMargin, y);
//     });
//   };

//   // Draw chart on the canvas
//   const drawData = (context, width, height, padding) => {
//     context.strokeStyle = color;
//     context.lineWidth = 2;

//     // Begin drawing the data
//     context.beginPath();
//     data.forEach((value, index) => {
//       const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
//       const y =
//         height -
//         padding -
//         ((value - minY) / (maxY - minY)) * (height - 2 * padding);

//       if (index === 0) {
//         context.moveTo(x, y);
//       } else {
//         context.lineTo(x, y);
//       }
//     });
//     context.stroke();
//   };

//   const drawHighlights = (context, width, height, padding) => {
//     highlightedSections.forEach((section) => {
//       const { start, end, color: sectionColor, label } = section;
//       const startX = padding + (start / data.length) * (width - 3 * padding);
//       const endX = padding + (end / data.length) * (width - 3 * padding);
//       const sectionWidth = endX - startX;

//       context.fillStyle = sectionColor;
//       context.globalAlpha = 0.3; // Make the highlight semi-transparent
//       context.fillRect(startX, padding, sectionWidth, height - 2 * padding);
//       context.globalAlpha = 1; // Reset alpha to default

//       // Draw section labels just above the highlight
//       context.fillStyle = color;
//       context.font = "12px Arial"; // Set font style for the label
//       context.textAlign = "center";

//       // Position label just above the highlighted section
//       const labelY = padding - 10; // 10 pixels above the top padding
//       context.fillText(label, startX + sectionWidth / 2, labelY);
//     });
//   };

//   // Base drawing (axes + data) stays mostly the same
//   const drawBase = () => {
//     const canvas = baseCanvasRef.current;
//     const context = canvas.getContext("2d");
//     const padding = 40;
//     const width = canvas.clientWidth;
//     const height = canvas.clientHeight;

//     context.clearRect(0, 0, width, height);
//     drawAxes(context, width, height, padding);
//     drawData(context, width, height, padding);
//   };

//   // Highlights only on overlay
//   const updateHighlights = () => {
//     const canvas = overlayCanvasRef.current;
//     const context = canvas.getContext("2d");
//     const padding = 40;
//     const width = canvas.clientWidth;
//     const height = canvas.clientHeight;

//     // Only clear the graph area, preserving axes
//     context.clearRect(padding, 0, width - padding, height - padding);
//     drawHighlights(context, width, height, padding);
//   };

//   // Base layer setup
//   useEffect(() => {
//     const handleResize = () => {
//       resizeCanvases();
//       drawBase();
//       updateHighlights();
//     };

//     resizeCanvases();
//     drawBase();

//     window.addEventListener("resize", handleResize);
//     return () => window.removeEventListener("resize", handleResize);
//   }, [data]);

//   // Highlight updates
//   useEffect(() => {
//     if (overlayCanvasRef.current) {
//       updateHighlights();
//     }
//   }, [highlightedSections]);

//   return (
//     <div className="relative w-full h-full">
//       <canvas
//         ref={baseCanvasRef}
//         className="absolute top-0 left-0 w-full h-full"
//       />
//       <canvas
//         ref={overlayCanvasRef}
//         className="absolute top-0 left-0 w-full h-full"
//       />
//     </div>
//   );
// };

// export default AudioFeaturesGraph;
