import { useEffect, useState, useCallback, useMemo } from "react";
import OverlayLineGraph from "./LineGraph/OverlayLineGraph";
import WaveformPlayer from "./WaveformPlayer";
import LoadingSpinner from "../LoadingSpinner";
import Tooltip from "../text/Tooltip";
import ToggleButton from "../buttons/ToggleButton";

const width = 800;
const graphHeight = 400;

const OverlayGraphWithWaveform = ({
  inputAudioURL,
  referenceAudioURL,
  inputFeatureData, // input audio feature data
  referenceFeatureData, // optional: reference performance feature data
  selectedAnalysisFeature,
  inputAudioDuration,
  referenceAudioDuration,
  tooltipMode,
  selectedModel,
  setSelectedModel,
}) => {
  const [selectedDataIndex, setSelectedDataIndex] = useState(0);
  const [chartState, setChartState] = useState(null);
  const emptyHighlightedSections = useMemo(() => [], []);

  const handleZoomChange = useCallback((changeData) => {
    setChartState(changeData);
  }, []);

  // Function to convert frame indices to time using your original formula
  const frameToTime = (frameIndex, inputAudioDuration, numFrames) => {
    return frameIndex * (inputAudioDuration / numFrames);
  };

  const referenceFrameToTime = (
    frameIndex,
    referenceAudioDuration,
    numFrames
  ) => {
    return frameIndex * (referenceAudioDuration / numFrames);
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

  // Active feature data
  const inputFeature = inputFeatureData?.[selectedDataIndex] || null;
  const referenceFeature = referenceFeatureData?.[selectedDataIndex] || null;

  const hasInputFeatureData =
    inputFeature &&
    Array.isArray(inputFeature.data) &&
    inputFeature.data.length > 0;

  const hasReference = !!(
    referenceFeature &&
    referenceFeature.data &&
    referenceFeature.data.length > 0
  );

  return (
    <div className="flex flex-col items-center justify-center w-full">
      {/* Reference Waveform player above */}
      <>
        {referenceAudioURL && (
          <div>
            <ul className="text-sm pb-2 pt-2">
              <li className="font-bold text-darkgray">reference audio</li>
            </ul>
            <WaveformPlayer
              feature={referenceFeature.label}
              key={referenceAudioURL}
              audioUrl={referenceAudioURL}
              highlightedSections={
                referenceFeature.highlighted?.audio?.length > 0
                  ? referenceFeature.highlighted.audio
                  : []
              }
              waveColor="#E0E0E0"
              progressColor="#A0A0A0"
              startTime={
                chartState?.zoom?.isZoomed && referenceAudioDuration
                  ? referenceFrameToTime(
                      chartState.zoom.startIndex,
                      referenceAudioDuration,
                      referenceFeature.data.length
                    )
                  : 0
              }
              endTime={
                chartState?.zoom?.isZoomed && referenceAudioDuration
                  ? referenceFrameToTime(
                      chartState.zoom.endIndex,
                      referenceAudioDuration,
                      referenceFeature.data.length
                    )
                  : referenceAudioDuration || undefined
              }
              audioDuration={referenceAudioDuration}
              playIconColorClass="text-darkgray"
              showTimeline={false}
            />
          </div>
        )}
      </>

      {!selectedAnalysisFeature ? (
        <div>Select an analysis feature above to start analyzing audio.</div>
      ) : inputFeatureData.length === 0 && selectedAnalysisFeature ? (
        <LoadingSpinner />
      ) : inputFeatureData === "invalid" ? (
        <div className="text-lightpink text-xl font-semibold">
          Not enough data to compute
        </div>
      ) : (
        inputFeature && (
          <>
            {/* Header and feature selector */}
            <div className="flex flex-row items-end w-full justify-between pt-4 mb-4">
              <ul className="text-sm text-lightgray">
                <li className="font-bold text-lightpink pt-2">
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
            <div className="flex flex-col items-center pb-6">
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
                  {/* Conditionally render OverlayLineGraph */}
                  {hasInputFeatureData && (
                    <OverlayLineGraph
                      key={`graph-${selectedDataIndex}`}
                      feature={inputFeature.label}
                      primaryData={inputFeature.data}
                      secondaryData={hasReference ? referenceFeature.data : []}
                      primaryLineColor="#FF89BB" // pink input
                      secondaryColor="#CCCCCC" // grey reference
                      width={width}
                      height={graphHeight}
                      xLabel="time (s)"
                      yLabel={
                        inputFeatureData[selectedDataIndex]?.label === "pitch"
                          ? "pitch (note)"
                          : inputFeatureData[selectedDataIndex]?.label ===
                            "dynamics"
                          ? "amplitude (dB)"
                          : inputFeatureData[selectedDataIndex]?.label ===
                              "rates" ||
                            inputFeatureData[selectedDataIndex]?.label ===
                              "extents"
                          ? "hz"
                          : inputFeatureData[selectedDataIndex]?.label ===
                            "tempo"
                          ? "beats per minute (bpm)"
                          : selectedAnalysisFeature === "phonation" ||
                            selectedAnalysisFeature === "vocal tone" ||
                            selectedAnalysisFeature === "pitch mod."
                          ? "probability"
                          : selectedAnalysisFeature
                      }
                      highlightedSections={
                        inputFeatureData[selectedDataIndex]?.highlighted
                          ?.data &&
                        inputFeatureData[selectedDataIndex]?.highlighted.data
                          .length > 0
                          ? inputFeatureData[selectedDataIndex]?.highlighted
                              ?.data
                          : emptyHighlightedSections
                      }
                      yMin={
                        inputFeatureData[selectedDataIndex]?.label === "pitch"
                          ? calculatePitchYMin(
                              inputFeatureData[selectedDataIndex].data
                            )
                          : inputFeatureData[selectedDataIndex]?.label ===
                            "tempo"
                          ? Math.max(
                              0,
                              Math.min(
                                ...inputFeatureData[selectedDataIndex].data
                              ) - 50
                            )
                          : selectedAnalysisFeature === "phonation" ||
                            selectedAnalysisFeature === "vocal tone" ||
                            selectedAnalysisFeature === "pitch mod."
                          ? 0
                          : Math.min(
                              ...inputFeatureData[selectedDataIndex].data
                            )
                      }
                      yMax={
                        inputFeatureData[selectedDataIndex]?.label ===
                          "tempo" ||
                        inputFeatureData[selectedDataIndex]?.label === "pitch"
                          ? Math.max(
                              ...inputFeatureData[selectedDataIndex].data
                            ) + 50
                          : selectedAnalysisFeature === "phonation" ||
                            selectedAnalysisFeature === "vocal tone" ||
                            selectedAnalysisFeature === "pitch mod."
                          ? 1
                          : Math.max(
                              ...inputFeatureData[selectedDataIndex].data
                            )
                      }
                      zoomDomain={
                        chartState?.zoom
                          ? [
                              chartState.zoom.startIndex,
                              chartState.zoom.endIndex,
                            ]
                          : null
                      }
                      onZoomChange={handleZoomChange}
                      style={{ position: "absolute", zIndex: 1 }}
                    />
                  )}
                </div>
              </Tooltip>

              {/* Input Audio Waveform player below */}
              <div>
                <ul className="text-sm pb-2 pt-2">
                  <li className="font-bold text-darkpink">performance audio</li>
                </ul>
                <WaveformPlayer
                  feature={inputFeature.label}
                  key={inputAudioURL}
                  audioUrl={inputAudioURL}
                  highlightedSections={
                    inputFeature.highlighted?.audio?.length > 0
                      ? inputFeature.highlighted.audio
                      : []
                  }
                  waveColor="#E0E0E0"
                  progressColor="#FF89BB"
                  startTime={
                    chartState?.zoom?.isZoomed && inputAudioDuration
                      ? frameToTime(
                          chartState.zoom.startIndex,
                          inputAudioDuration,
                          inputFeature.data.length
                        )
                      : 0
                  }
                  endTime={
                    chartState?.zoom?.isZoomed && inputAudioDuration
                      ? frameToTime(
                          chartState.zoom.endIndex,
                          inputAudioDuration,
                          inputFeature.data.length
                        )
                      : inputAudioDuration || undefined
                  }
                  audioDuration={inputAudioDuration}
                  showTimeline={false}
                />
              </div>
            </div>
            {["vocal tone", "pitch mod."].includes(
              selectedAnalysisFeature?.toLowerCase()
            ) && (
              <div className="flex justify-end w-full">
                <ToggleButton
                  question="Select Model:"
                  options={["CLAP", "Whisper"]}
                  allowOther={false}
                  background_color="bg-white/10"
                  onChange={(selected) => setSelectedModel(selected)}
                  isMultiSelect={false}
                  showToggle={false}
                  miniVersion={true}
                  selected={selectedModel}
                />
              </div>
            )}
          </>
        )
      )}
    </div>
  );
};

export default OverlayGraphWithWaveform;
