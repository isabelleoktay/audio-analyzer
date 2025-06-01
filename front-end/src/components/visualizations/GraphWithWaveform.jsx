import { useState, useEffect } from "react";
import LineGraph from "./LineGraph";
import WaveformPlayer from "./WaveformPlayer";
import LoadingSpinner from "../LoadingSpinner";

const width = 800;
// const leftMargin = 50;
// const rightMargin = 20;
// const buttonWidth = 40; // w-10 = 40px
const graphHeight = 400;
// const waveformHeight = 50;

const GraphWithWaveform = ({
  audioURL,
  featureData,
  highlightedDataSection,
  highlightedAudioSection,
  selectedAnalysisFeature,
  xLabels,
}) => {
  const [loading, setLoading] = useState(false);

  console.log(audioURL);

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
            feature={selectedAnalysisFeature}
            data={featureData}
            width={width}
            height={graphHeight}
            xLabel="time (s)"
            yLabel={
              selectedAnalysisFeature === "pitch"
                ? "pitch (note)"
                : selectedAnalysisFeature
            }
            highlightedSections={
              highlightedDataSection &&
              highlightedDataSection.start !== undefined &&
              highlightedDataSection.end !== undefined
                ? [highlightedDataSection]
                : []
            }
            yMin={
              selectedAnalysisFeature === "tempo" ||
              selectedAnalysisFeature === "pitch"
                ? Math.max(0, Math.min(...featureData) - 50)
                : Math.min(...featureData)
            }
            yMax={
              selectedAnalysisFeature === "tempo" ||
              selectedAnalysisFeature === "pitch"
                ? Math.max(...featureData) + 50
                : Math.max(...featureData)
            }
            xLabels={xLabels || []}
          />
        )}
      </div>
      <WaveformPlayer
        key={audioURL}
        audioUrl={audioURL}
        // width={width - leftMargin - rightMargin}
        // height={waveformHeight}
        highlightedSections={
          highlightedAudioSection &&
          highlightedAudioSection.start !== undefined &&
          highlightedAudioSection.end !== undefined
            ? [highlightedAudioSection]
            : []
        }
        // wavesurferRef={wavesurferRef}
        // setIsPlaying={setIsPlaying}
        waveColor="#E0E0E0"
        progressColor="#90F1EF"
      />
    </div>
  );
};

export default GraphWithWaveform;
