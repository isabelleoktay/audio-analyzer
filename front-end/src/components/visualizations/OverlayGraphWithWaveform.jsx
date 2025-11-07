import { useEffect, useState, useCallback, useMemo } from "react";
import OverlayLineGraph from "./LineGraph/OverlayLineGraph";
import WaveformPlayer from "./WaveformPlayer";
import LoadingSpinner from "../LoadingSpinner";
import Tooltip from "../text/Tooltip";

import { mockInputFeatures, mockReferenceFeatures } from "../../mock/index";

const width = 800;
const graphHeight = 400;

const OverlayGraphWithWaveform = ({
  inputAudioURL,
  referenceAudioURL,
  inputFeatureData = mockInputFeatures, // input audio feature data
  referenceFeatureData = mockReferenceFeatures, // optional: reference performance feature data
  selectedAnalysisFeature,
  selectedVoiceType,
  selectedModel,
  inputAudioDuration,
  referenceAudioDuration,
  tooltipMode,
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
    const positiveValues = data.filter((v) => v > 0);
    if (positiveValues.length === 0) return 0;
    return Math.max(0, Math.min(...positiveValues) - 10);
  };

  // Active feature data
  const modelInputData = inputFeatureData?.[selectedModel] || inputFeatureData;
  const modelReferenceData =
    referenceFeatureData?.[selectedModel] || referenceFeatureData;

  const inputFeature = modelInputData?.[selectedDataIndex] || null;
  const referenceFeature = modelReferenceData?.[selectedDataIndex] || null;

  const hasInputFeatureData =
    inputFeature &&
    Array.isArray(inputFeature.data) &&
    inputFeature.data.length > 0;

  const hasReference = !!(
    referenceFeature &&
    referenceFeature.data &&
    referenceFeature.data.length > 0
  );

  // Compute yMin/yMax safely based on whichever datasets exist
  const safeData = (arr) =>
    Array.isArray(arr) ? arr.filter((d) => d !== null && !isNaN(d)) : [];

  const yMin = (() => {
    if (!hasInputFeatureData) return 0;
    if (hasReference && referenceFeature?.data) {
      return Math.min(
        ...safeData(inputFeature.data),
        ...safeData(referenceFeature.data)
      );
    }
    return Math.min(...safeData(inputFeature.data));
  })();

  const yMax = (() => {
    if (!hasInputFeatureData) return 1;
    if (hasReference && referenceFeature?.data) {
      return Math.max(
        ...safeData(inputFeature.data),
        ...safeData(referenceFeature.data)
      );
    }
    return Math.max(...safeData(inputFeature.data));
  })();

  return (
    <div className="flex flex-col items-center justify-center w-full">
      {/* Reference Waveform player above */}
      <>
        {referenceAudioURL && (
          <WaveformPlayer
            feature={inputFeature.label}
            key={referenceAudioURL}
            audioUrl={referenceAudioURL}
            highlightedSections={
              inputFeature.highlighted?.audio?.length > 0
                ? inputFeature.highlighted.audio
                : []
            }
            waveColor="#E0E0E0"
            progressColor="#A0A0A0"
            startTime={
              chartState?.zoom?.isZoomed && referenceAudioDuration
                ? referenceFrameToTime(
                    chartState.zoom.startIndex,
                    referenceAudioDuration,
                    inputFeature.data.length
                  )
                : 0
            }
            endTime={
              chartState?.zoom?.isZoomed && referenceAudioDuration
                ? referenceFrameToTime(
                    chartState.zoom.endIndex,
                    referenceAudioDuration,
                    inputFeature.data.length
                  )
                : referenceAudioDuration || undefined
            }
            audioDuration={referenceAudioDuration}
            playIconColorClass="text-darkgray"
            showTimeline={false}
          />
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
                      onZoomChange={handleZoomChange}
                      style={{ position: "absolute", zIndex: 1 }}
                    />
                  )}
                </div>
              </Tooltip>

              {/* Input Audio Waveform player below */}
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
          </>
        )
      )}
    </div>
  );
};

export default OverlayGraphWithWaveform;
