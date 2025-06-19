import { useEffect, useState, useCallback } from "react";
import LineGraph from "./LineGraph/LineGraph";
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
  selectedAnalysisFeature,
  audioDuration,
}) => {
  const [selectedDataIndex, setSelectedDataIndex] = useState(0);
  const [chartState, setChartState] = useState(null);

  const handleZoomChange = useCallback((changeData) => {
    setChartState(changeData);

    console.log("Chart updated:", {
      zoom: changeData.zoom,
      isZoomed: changeData.zoom.isZoomed,
    });
  }, []);

  // Function to convert frame indices to time using your original formula
  const frameToTime = (frameIndex, audioDuration, numFrames) => {
    return frameIndex * (audioDuration / numFrames);
  };

  const handleButtonClick = (index) => {
    setSelectedDataIndex(index);
  };

  useEffect(() => {
    setSelectedDataIndex(0);
  }, [selectedAnalysisFeature]);

  const calculatePitchYMin = (data) => {
    // Filter out values that are 0 or negative
    const positiveValues = data.filter((value) => value > 0);

    if (positiveValues.length === 0) {
      return 0; // Fallback if no positive values
    }

    const minPositiveValue = Math.min(...positiveValues);
    return Math.max(0, minPositiveValue - 10);
  };

  return (
    <div
      className="flex flex-col items-center justify-center"
      style={{ width, height: graphHeight + 100 }}
    >
      {!selectedAnalysisFeature ? (
        <div>select an analysis feature above to start analyzing audio </div>
      ) : featureData.length === 0 && selectedAnalysisFeature ? (
        <LoadingSpinner />
      ) : (
        featureData &&
        featureData[selectedDataIndex] && (
          <>
            {/* Render buttons for each data object */}
            <div className="flex space-x-4 self-end">
              {featureData?.map((d, index) => (
                <div
                  key={index}
                  onClick={() => handleButtonClick(index)}
                  className={`text-sm cursor-pointer ${
                    selectedDataIndex === index
                      ? "text-electricblue"
                      : "text-lightgray/50"
                  }`}
                >
                  {d.label}
                </div>
              ))}
            </div>

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
                <LineGraph
                  feature={featureData[selectedDataIndex]?.label}
                  data={featureData[selectedDataIndex]?.data}
                  lineColor={
                    featureData[selectedDataIndex].label === "extents" ||
                    featureData[selectedDataIndex].label === "rates"
                      ? "#E0E0E0"
                      : "#FF89BB"
                  }
                  width={width}
                  height={graphHeight}
                  xLabel="time (s)"
                  yLabel={
                    featureData[selectedDataIndex]?.label === "pitch" ||
                    featureData[selectedDataIndex]?.label === "vibrato"
                      ? "pitch (note)"
                      : featureData[selectedDataIndex]?.label === "dynamics"
                      ? "amplitude (dB)"
                      : featureData[selectedDataIndex]?.label === "rates" ||
                        featureData[selectedDataIndex]?.label === "extents"
                      ? "hz"
                      : featureData[selectedDataIndex]?.label === "tempo"
                      ? "beats per minute (bpm)"
                      : selectedAnalysisFeature
                  }
                  highlightedSections={
                    featureData[selectedDataIndex]?.highlighted?.data &&
                    featureData[selectedDataIndex]?.highlighted.data.length > 0
                      ? featureData[selectedDataIndex]?.highlighted?.data
                      : []
                  }
                  onZoomChange={handleZoomChange}
                  yMin={
                    featureData[selectedDataIndex]?.label === "pitch" ||
                    featureData[selectedDataIndex]?.label === "vibrato"
                      ? calculatePitchYMin(featureData[selectedDataIndex].data)
                      : featureData[selectedDataIndex]?.label === "tempo"
                      ? Math.max(
                          0,
                          Math.min(...featureData[selectedDataIndex].data) - 50
                        )
                      : selectedAnalysisFeature === "phonation"
                      ? 0
                      : Math.min(...featureData[selectedDataIndex].data)
                  }
                  yMax={
                    featureData[selectedDataIndex]?.label === "tempo" ||
                    featureData[selectedDataIndex]?.label === "pitch" ||
                    featureData[selectedDataIndex]?.label === "vibrato"
                      ? Math.max(...featureData[selectedDataIndex].data) + 50
                      : selectedAnalysisFeature === "phonation"
                      ? 1
                      : Math.max(...featureData[selectedDataIndex].data)
                  }
                />
              </div>
              <WaveformPlayer
                key={audioURL}
                audioUrl={audioURL}
                // width={width - leftMargin - rightMargin}
                // height={waveformHeight}
                highlightedSections={
                  featureData[selectedDataIndex]?.highlighted?.audio &&
                  featureData[selectedDataIndex].highlighted.audio.length > 0
                    ? featureData[selectedDataIndex]?.highlighted?.audio
                    : []
                }
                waveColor="#E0E0E0"
                progressColor="#90F1EF"
                startTime={
                  chartState?.zoom?.isZoomed &&
                  audioDuration &&
                  featureData[selectedDataIndex]
                    ? frameToTime(
                        chartState.zoom.startIndex,
                        audioDuration,
                        featureData[selectedDataIndex].data.length
                      )
                    : 0
                }
                endTime={
                  chartState?.zoom?.isZoomed &&
                  audioDuration &&
                  featureData[selectedDataIndex]
                    ? frameToTime(
                        chartState.zoom.endIndex,
                        audioDuration,
                        featureData[selectedDataIndex].data.length
                      )
                    : audioDuration || undefined
                }
                audioDuration={audioDuration}
              />
            </div>
          </>
        )
      )}
    </div>
  );
};

export default GraphWithWaveform;
