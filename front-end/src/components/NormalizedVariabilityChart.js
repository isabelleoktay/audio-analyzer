import React, { useRef, useEffect } from "react";
import SquareIcon from "@mui/icons-material/Square";

const NormalizedVariabilityChart = ({
  timbre,
  loudness,
  pitch,
  articulation,
  xTicks,
  tickInterval = 10, // Add a new prop to control the interval of displayed ticks
}) => {
  const canvasRef = useRef(null);
  const features = [
    { data: timbre, color: "#ff0000", label: "Timbre" },
    { data: loudness, color: "#008000", label: "Loudness" },
    { data: pitch, color: "#FFA500", label: "Pitch Difference" },
    { data: articulation, color: "#800080", label: "Articulation Level" },
  ];

  useEffect(() => {
    const margin = { top: 20, right: 15, bottom: 40, left: 30 };
    const width = 750 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = (width + margin.left + margin.right) * dpr;
    canvas.height = (height + margin.top + margin.bottom) * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(margin.left, margin.top);

    const dataLength = timbre.length;
    const xScale = (index) => (index / (dataLength - 1)) * width;

    // Draw each feature line
    features.forEach(({ data, color }) => {
      ctx.beginPath();
      ctx.moveTo(0, (1 - data[0]) * height); // Invert Y-axis for drawing
      data.forEach((value, i) => {
        ctx.lineTo(xScale(i), (1 - value) * height);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Draw axes
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(width, height); // X-axis
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, height); // Y-axis
    ctx.stroke();

    // Draw x-axis labels using provided xTicks
    const numTicks = xTicks.length;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#1e293b";
    xTicks.forEach((tick, i) => {
      if (i % tickInterval === 0) {
        // Display every nth tick
        const x = (i / (numTicks - 1)) * width;
        ctx.fillText(parseFloat(tick).toFixed(2), x, height + 5);
      }
    });

    // Draw x-axis label
    ctx.fillText("Time (seconds)", width / 2, height + 25);

    // Draw y-axis labels
    const yTicks = 5;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= yTicks; i++) {
      const y = (i / yTicks) * height;
      const value = (1 - i / yTicks).toFixed(2);
      ctx.fillText(value, -5, y);
    }

    // Reset translation
    ctx.translate(-margin.left, -margin.top);
  }, [timbre, loudness, pitch, articulation, xTicks, tickInterval]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-center mb-4 text-slate-800">
        Normalized Variablity Over Time
      </div>
      <div className="flex flex-row items-center">
        <canvas ref={canvasRef} className="w-full h-56" />
        <div className="flex flex-col justify-center ml-4 self-center">
          {features.map((feature, idx) => (
            <div key={idx} className="flex items-center space-x-2">
              <SquareIcon style={{ color: feature.color }} />
              <span className="text-sm text-slate-800">{feature.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default NormalizedVariabilityChart;
