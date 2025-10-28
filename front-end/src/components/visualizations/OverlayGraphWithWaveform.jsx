import { useEffect, useState, useCallback, useMemo } from "react";
import LineGraph from "./LineGraph/LineGraph";
import WaveformPlayer from "./WaveformPlayer";
import LoadingSpinner from "../LoadingSpinner";
import Tooltip from "../text/Tooltip";

const width = 800;
const graphHeight = 400;

const OverlayGraphWithWaveform = ({
  inputAudioURL,
  referenceAudioURL,
  inputFeatureData, // input audio feature data
  referenceFeatureData, // new: reference performance feature data
  selectedAnalysisFeature,
  inputAudioDuration,
  tooltipMode,
}) => {
  const [selectedDataIndex, setSelectedDataIndex] = useState(0);
  const [chartState, setChartState] = useState(null);
  const emptyHighlightedSections = useMemo(() => [], []);

  const handleZoomChange = useCallback((changeData) => {
    setChartState(changeData);
  }, []);

  // Convert frame index to time
  const frameToTime = (frameIndex, duration, numFrames) => {
    return frameIndex * (duration / numFrames);
  };

  const handleButtonClick = (index) => {
    setSelectedDataIndex(index);
  };

  useEffect(() => {
    setSelectedDataIndex(0);
  }, [selectedAnalysisFeature]);

  const calculatePitchYMin = (data) => {
    const positiveValues = data.filter((v) => v > 0);
    if (positiveValues.length === 0) return 0;
    return Math.max(0, Math.min(...positiveValues) - 10);
  };

  const activeFeature = inputFeatureData[selectedDataIndex];
  const referenceFeature = referenceFeatureData?.[selectedDataIndex];

  return (
    <div
      className="flex flex-col items-center justify-center w-full"
      style={{ width, height: graphHeight + 100 }}
    >
      {!selectedAnalysisFeature ? (
        <div>Select an analysis feature above to start analyzing audio.</div>
      ) : inputFeatureData.length === 0 && selectedAnalysisFeature ? (
        <LoadingSpinner />
      ) : inputFeatureData === "invalid" ? (
        <div className="text-lightpink text-xl font-semibold">
          Not enough data to compute
        </div>
      ) : (
        activeFeature && (
          <>
            {/* Header and feature selector */}
            <div className="flex flex-row items-end w-full justify-between">
              <ul className="text-sm text-lightgray">
                <li className="font-bold text-lightpink">
                  Click and drag on the graph area to zoom in!
                </li>
              </ul>
              <div className="flex space-x-4 self-end">
                {inputFeatureData?.map((d, index) => (
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
            </div>

            {/* Graph */}
            <div className="flex flex-col items-center">
              <Tooltip
                text="click and drag to zoom in on the graph"
                position="top"
                show={tooltipMode === "global"}
                tooltipMode={tooltipMode}
              >
                <div
                  className="w-full flex items-center justify-center"
                  style={{ height: graphHeight }}
                >
                  {/* Reference LineGraph in gray */}
                  {referenceFeature && (
                    <LineGraph
                      key={`reference-${selectedDataIndex}`}
                      feature={referenceFeature.label}
                      data={referenceFeature.data}
                      lineColor="#CCCCCC" // grey reference
                      width={width}
                      height={graphHeight}
                      xLabel="time (s)"
                      yLabel={selectedAnalysisFeature}
                      highlightedSections={emptyHighlightedSections}
                      onZoomChange={handleZoomChange}
                      yMin={Math.min(...referenceFeature.data)}
                      yMax={Math.max(...referenceFeature.data)}
                      style={{ position: "absolute", zIndex: 1, opacity: 0.7 }}
                    />
                  )}

                  {/* Input LineGraph in pink */}
                  <LineGraph
                    key={`input-${selectedDataIndex}`}
                    feature={activeFeature.label}
                    data={activeFeature.data}
                    lineColor="#FF89BB" // pink input overlay
                    width={width}
                    height={graphHeight}
                    xLabel="time (s)"
                    yLabel={selectedAnalysisFeature}
                    highlightedSections={
                      activeFeature.highlighted?.data?.length > 0
                        ? activeFeature.highlighted.data
                        : emptyHighlightedSections
                    }
                    onZoomChange={handleZoomChange}
                    yMin={calculatePitchYMin(activeFeature.data)}
                    yMax={Math.max(...activeFeature.data) + 50}
                    style={{ position: "absolute", zIndex: 2 }}
                  />
                </div>
              </Tooltip>

              {/* Waveform player below */}
              <WaveformPlayer
                feature={activeFeature.label}
                key={inputAudioURL}
                inputAudioURL={inputAudioURL}
                highlightedSections={
                  activeFeature.highlighted?.audio?.length > 0
                    ? activeFeature.highlighted.audio
                    : []
                }
                waveColor="#E0E0E0"
                progressColor="#90F1EF"
                startTime={
                  chartState?.zoom?.isZoomed && inputAudioDuration
                    ? frameToTime(
                        chartState.zoom.startIndex,
                        inputAudioDuration,
                        activeFeature.data.length
                      )
                    : 0
                }
                endTime={
                  chartState?.zoom?.isZoomed && inputAudioDuration
                    ? frameToTime(
                        chartState.zoom.endIndex,
                        inputAudioDuration,
                        activeFeature.data.length
                      )
                    : inputAudioDuration || undefined
                }
                inputAudioDuration={inputAudioDuration}
              />
            </div>
          </>
        )
      )}
    </div>
  );
};

export default OverlayGraphWithWaveform;
