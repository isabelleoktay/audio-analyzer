import { useRef, useState, useEffect } from "react";
import LineGraph from "./LineGraph";
import WaveformPlayer from "./WaveformPlayer";
import { FaPlay, FaPause } from "react-icons/fa";
import IconButton from "../buttons/IconButton";
import LoadingSpinner from "../LoadingSpinner";

const width = 800;
const leftMargin = 50;
const rightMargin = 20;
const buttonWidth = 40; // w-10 = 40px
const graphHeight = 400;
const waveformHeight = 50;

const GraphWithWaveform = ({
  audioURL,
  featureData,
  sampleRate,
  highlightedSections,
  selectedAnalysisFeature,
}) => {
  const wavesurferRef = useRef();
  const [isPlaying, setIsPlaying] = useState(false);
  const [loading, setLoading] = useState(false);

  const handlePlayPause = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
    setIsPlaying((prev) => !prev);
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Prevent default space bar scrolling
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        handlePlayPause();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    const isLoading =
      !featureData || !Array.isArray(featureData) || featureData.length === 0;
    setLoading(isLoading);
  }, [featureData]);

  return (
    <div className="flex flex-col items-center" style={{ width }}>
      <div
        style={{
          height: graphHeight,
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {loading ? (
          <LoadingSpinner />
        ) : (
          <LineGraph
            data={featureData}
            width={width}
            height={graphHeight}
            xLabel="time (s)"
            yLabel={selectedAnalysisFeature}
            highlightedSections={highlightedSections}
            yMin={
              selectedAnalysisFeature === "tempo"
                ? Math.max(0, Math.min(...featureData) - 50)
                : Math.min(...featureData)
            }
            yMax={
              selectedAnalysisFeature === "tempo"
                ? Math.max(...featureData) + 50
                : Math.max(...featureData)
            }
          />
        )}
      </div>
      <div className="flex flex-row items-center w-full">
        <div
          style={{
            width: buttonWidth,
            display: "flex",
            justifyContent: "center",
          }}
        >
          <IconButton
            icon={isPlaying ? FaPause : FaPlay}
            onClick={handlePlayPause}
            colorClass="text-darkpink"
            bgClass="bg-transparent"
            sizeClass="w-10 h-10"
            ariaLabel="play audio"
          />
        </div>
        <div
          style={{
            width: width - leftMargin - rightMargin,
            marginLeft: leftMargin - buttonWidth,
            marginRight: rightMargin,
            boxSizing: "border-box",
          }}
        >
          <WaveformPlayer
            key={audioURL}
            audioUrl={audioURL}
            width={width - leftMargin - rightMargin}
            height={waveformHeight}
            highlightedSections={highlightedSections}
            wavesurferRef={wavesurferRef}
            setIsPlaying={setIsPlaying}
          />
        </div>
      </div>
    </div>
  );
};

export default GraphWithWaveform;
