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
  similarityScore,
  setSimilarityScore,
}) => {
  const [selectedDataIndex, setSelectedDataIndex] = useState(0);
  const [chartState, setChartState] = useState(null);

  // Filter inputFeatureData based on selectedModel if it's a model-based feature
  const displayedInputFeatureData = useMemo(() => {
    if (!inputFeatureData) return [];

    // Check if data has model keys (CLAP, Whisper, etc.)
    if (
      typeof inputFeatureData === "object" &&
      !Array.isArray(inputFeatureData) &&
      (inputFeatureData.CLAP || inputFeatureData.Whisper)
    ) {
      // Model-based structure: return data for selected model
      return inputFeatureData[selectedModel] || [];
    }

    // Flat array structure: return as-is
    return Array.isArray(inputFeatureData) ? inputFeatureData : [];
  }, [inputFeatureData, selectedModel]);

  // Filter referenceFeatureData based on selectedModel if it's a model-based feature
  const displayedReferenceFeatureData = useMemo(() => {
    if (!referenceFeatureData) return [];

    // Check if data has model keys
    if (
      typeof referenceFeatureData === "object" &&
      !Array.isArray(referenceFeatureData) &&
      (referenceFeatureData.CLAP || referenceFeatureData.Whisper)
    ) {
      // Model-based structure: return data for selected model
      return referenceFeatureData[selectedModel] || [];
    }

    // Flat array structure: return as-is
    return Array.isArray(referenceFeatureData) ? referenceFeatureData : [];
  }, [referenceFeatureData, selectedModel]);
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
    numFrames,
  ) => {
    return frameIndex * (referenceAudioDuration / numFrames);
  };

  const handleButtonClick = (index) => {
    setSelectedDataIndex(index);
  };

  useEffect(() => {
    setSelectedDataIndex(0);
  }, [selectedAnalysisFeature]);

  //   useEffect(() => {
  //     console.log(inputFeatureData);
  //   }, [inputFeatureData]);

  const calculatePitchYMin = (data) => {
    // Filter out values that are 0 or negative
    const positiveValues = data.filter((value) => value > 0);
    if (positiveValues.length === 0) {
      return 0; // Fallback if no positive values
    }
    const minPositiveValue = Math.min(...positiveValues);
    return Math.max(0, minPositiveValue - 10);
  };

  // --- added: combine input + reference values for global min/max calculation ---
  const combinedValuesForIndex = (index) => {
    const inputVals = displayedInputFeatureData?.[index]?.data || [];
    const refVals = displayedReferenceFeatureData?.[index]?.data || [];
    return [...inputVals, ...refVals].filter(
      (v) => typeof v === "number" && !isNaN(v),
    );
  };

  const computeYBounds = (index) => {
    const label = displayedInputFeatureData?.[index]?.label;
    const combined = combinedValuesForIndex(index);
    const hasValues = combined.length > 0;
    const globalMin = hasValues ? Math.min(...combined) : 0;
    const globalMax = hasValues ? Math.max(...combined) : 1;

    // yMin
    let yMin;
    if (label === "pitch") {
      yMin = calculatePitchYMin(combined.length ? combined : [0]);
    } else if (label === "tempo") {
      yMin = Math.max(0, globalMin - 50);
    } else if (
      selectedAnalysisFeature === "phonation" ||
      selectedAnalysisFeature === "vocal tone" ||
      selectedAnalysisFeature === "pitch mod."
    ) {
      yMin = 0;
    } else {
      yMin = globalMin;
    }

    // yMax
    let yMax;
    if (label === "tempo" || label === "pitch") {
      yMax = globalMax + 50;
    } else if (
      selectedAnalysisFeature === "phonation" ||
      selectedAnalysisFeature === "vocal tone" ||
      selectedAnalysisFeature === "pitch mod."
    ) {
      yMax = 1;
    } else {
      yMax = globalMax;
    }

    return { yMin, yMax };
  };

  // Use computed bounds for the currently selected index
  const { yMin, yMax } = computeYBounds(selectedDataIndex);

  // Active feature data

  const inputFeature =
    displayedInputFeatureData === "invalid"
      ? null
      : displayedInputFeatureData?.[selectedDataIndex];
  const referenceFeature =
    displayedReferenceFeatureData === "invalid"
      ? null
      : displayedReferenceFeatureData?.[selectedDataIndex];

  const hasInputFeatureData =
    inputFeature &&
    Array.isArray(inputFeature.data) &&
    inputFeature.data.length > 0;

  const hasReference = !!(
    referenceFeature &&
    referenceFeature.data &&
    referenceFeature.data.length > 0
  );

  const hasReferenceFile = Boolean(referenceAudioURL);

  return (
    <div className="flex flex-col items-center justify-center w-full">
      {/* Reference Waveform player above */}
      {hasInputFeatureData && (
        <>
          {hasReferenceFile ? (
            referenceFeature ? (
              <div>
                <ul className="text-sm pb-2 pt-2">
                  <li className="font-bold text-darkgray">reference audio</li>
                </ul>
                <WaveformPlayer
                  feature={referenceFeature?.label}
                  key={referenceAudioURL}
                  audioUrl={referenceAudioURL}
                  //   highlightedSections={
                  //     referenceFeature.highlighted?.audio?.length > 0
                  //       ? referenceFeature.highlighted.audio
                  //       : []
                  //   }
                  waveColor="#E0E0E0"
                  progressColor="#A0A0A0"
                  startTime={
                    chartState?.zoom?.isZoomed && referenceAudioDuration
                      ? referenceFrameToTime(
                          chartState.zoom.startIndex,
                          referenceAudioDuration,
                          referenceFeature.data.length,
                        )
                      : 0
                  }
                  endTime={
                    chartState?.zoom?.isZoomed && referenceAudioDuration
                      ? referenceFrameToTime(
                          chartState.zoom.endIndex,
                          referenceAudioDuration,
                          referenceFeature.data.length,
                        )
                      : referenceAudioDuration || undefined
                  }
                  audioDuration={referenceAudioDuration}
                  playIconColorClass="text-darkgray"
                  showTimeline={false}
                />
              </div>
            ) : (
              <div className="text-lightpink text-xl font-semibold">
                Not able to compute feature for provided reference file.
              </div>
            )
          ) : null}
          {/* No reference file → render nothing */}
        </>
      )}

      {!selectedAnalysisFeature ? (
        <div>Select an analysis feature above to start analyzing audio.</div>
      ) : (typeof inputFeatureData === "object" &&
        !Array.isArray(inputFeatureData)
          ? Object.keys(inputFeatureData).length === 0
          : displayedInputFeatureData.length === 0) &&
        selectedAnalysisFeature ? (
        <LoadingSpinner />
      ) : inputFeatureData === "invalid" ? (
        <div className="text-lightpink text-xl font-semibold">
          Not able to compute feature for provided input file.
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
                {displayedInputFeatureData?.map((d, index) => (
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
                      //   highlightedSections={
                      //     inputFeatureData[selectedDataIndex]?.highlighted
                      //       ?.data &&
                      //     inputFeatureData[selectedDataIndex]?.highlighted.data
                      //       .length > 0
                      //       ? inputFeatureData[selectedDataIndex]?.highlighted
                      //           ?.data
                      //       : emptyHighlightedSections
                      //   }
                      yMin={yMin}
                      yMax={yMax}
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
                      similarityScore={similarityScore}
                      onSimilarityCalculated={(score) =>
                        setSimilarityScore(score)
                      }
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
                  // highlightedSections={
                  //   inputFeature.highlighted?.audio?.length > 0
                  //     ? inputFeature.highlighted.audio
                  //     : []
                  // }
                  waveColor="#E0E0E0"
                  progressColor="#FF89BB"
                  startTime={
                    chartState?.zoom?.isZoomed && inputAudioDuration
                      ? frameToTime(
                          chartState.zoom.startIndex,
                          inputAudioDuration,
                          inputFeature.data.length,
                        )
                      : 0
                  }
                  endTime={
                    chartState?.zoom?.isZoomed && inputAudioDuration
                      ? frameToTime(
                          chartState.zoom.endIndex,
                          inputAudioDuration,
                          inputFeature.data.length,
                        )
                      : inputAudioDuration || undefined
                  }
                  audioDuration={inputAudioDuration}
                  showTimeline={false}
                />
              </div>
            </div>
            {["vocal tone", "pitch mod."].includes(
              selectedAnalysisFeature?.toLowerCase(),
            ) && (
              <div className="flex justify-end w-full">
                {(() => {
                  // Check if original data has both models
                  const hasClap =
                    Array.isArray(inputFeatureData?.CLAP) &&
                    inputFeatureData.CLAP.length > 0;
                  const hasWhisper =
                    Array.isArray(inputFeatureData?.Whisper) &&
                    inputFeatureData.Whisper.length > 0;
                  const availableModels = [];
                  if (hasClap) availableModels.push("CLAP");
                  if (hasWhisper) availableModels.push("Whisper");

                  // Only show toggle if we have more than one model
                  if (availableModels.length > 1) {
                    return (
                      <ToggleButton
                        question="Select Model:"
                        options={availableModels}
                        allowOther={false}
                        background_color="bg-white/10"
                        onChange={(selected) => setSelectedModel(selected)}
                        isMultiSelect={false}
                        showToggle={false}
                        miniVersion={true}
                        selected={selectedModel}
                      />
                    );
                  }
                  return null;
                })()}
              </div>
            )}
          </>
        )
      )}
    </div>
  );
};

export default OverlayGraphWithWaveform;
