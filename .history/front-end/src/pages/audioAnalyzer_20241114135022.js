import React, { useState, useRef, useEffect } from "react";
import Header from "../components/Header";
import FileUploader from "../components/FileUploader";
import WaveformPlayback from "../components/WaveformPlayback";
import AudioFeaturesDisplay from "../components/AudioFeaturesDisplay";
import Tabs from "../components/Tabs";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import AudioFeaturesGraph from "../components/AudioFeaturesGraph";

const AudioAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [waveformAxes, setWaveformAxes] = useState(null);
  const [variabilityAxes, setVariabilityAxes] = useState(null);
  const [highlightedSections, setHighlightedSections] = useState(null);
  const [selectedHighlightedSections, setSelectedHighlightedSections] =
    useState([]);
  const [playingSection, setPlayingSection] = useState(null);
  const [features, setFeatures] = useState(null);
  const [activeFeatureTab, setActiveFeatureTab] = useState("Loudness");
  const [activeVisualizationTab, setActiveVisualizationTab] =
    useState("Highlights");
  const [playingAudioRange, setPlayingAudioRange] = useState(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);

  const visualizationTabs = [
    { name: "Highlights", color: "blue-500" },
    { name: "Variability", color: "blue-500" },
  ];

  const featureTabs = [
    { name: "Loudness", color: "blue-500" },
    { name: "Pitch", color: "blue-500" },
  ];

  const calculateAxes = (data, sampleRate, hopLength) => {
    const minY = Math.min(...data);
    const maxY = Math.max(...data);

    // X-axis time labels based on sample rate and hop length
    const duration = (data.length * hopLength) / sampleRate; // total time in seconds
    const xLabels = Array.from({ length: 6 }, (_, i) => ({
      label: `${((duration * i) / 5).toFixed(2)}s`,
      position: i / 5, // normalized position (0 to 1) to use within width
    }));

    // Y-axis labels based on min and max values
    const yLabels = Array.from({ length: 5 }, (_, i) => ({
      label: (minY + ((maxY - minY) * i) / 4).toFixed(2),
      position: i / 4, // normalized (1 to 0) for top to bottom
    }));

    return { xLabels, yLabels, minY, maxY };
  };

  const calculateWaveformAxes = (data) => {
    // Calculate min and max values for the y-axis
    const minY = data.reduce((min, val) => Math.min(min, val), Infinity);
    const maxY = data.reduce((max, val) => Math.max(max, val), -Infinity);

    // Calculate total duration in seconds (number of samples / sample rate)
    const duration = audioBuffer.duration;

    // Calculate x-axis labels (time in seconds)
    const xLabels = Array.from({ length: 6 }, (_, i) => ({
      label: `${((duration * i) / 5).toFixed(2)}s`, // Time label at specific intervals
      position: i / 5, // Normalized position (0 to 1) to fit within the canvas width
    }));

    // Calculate y-axis labels (min to max amplitude/frequency)
    const yLabels = Array.from({ length: 5 }, (_, i) => ({
      label: (minY + ((maxY - minY) * i) / 4).toFixed(2), // Label range between minY and maxY
      position: i / 4, // Normalized position (0 to 1) for vertical placement on canvas (bottom to top)
    }));

    return { xLabels, yLabels, minY, maxY };
  };

  const calculateVariabilityAxes = (data) => {
    // Calculate min and max values for the y-axis
    const minY = data.reduce((min, val) => Math.min(min, val), Infinity);
    const maxY = data.reduce((max, val) => Math.max(max, val), -Infinity);

    // Calculate total duration in seconds (number of samples / sample rate)
    const duration = audioBuffer.duration;

    // Calculate x-axis labels (time in seconds)
    const xLabels = Array.from({ length: 6 }, (_, i) => ({
      label: `${((duration * i) / 5).toFixed(2)}s`, // Time label at specific intervals
      position: i / 5, // Normalized position (0 to 1) to fit within the canvas width
    }));

    // Calculate y-axis labels (min to max amplitude/frequency)
    const yLabels = Array.from({ length: 5 }, (_, i) => ({
      label: (minY + ((maxY - minY) * i) / 4).toFixed(2), // Label range between minY and maxY
      position: i / 4, // Normalized position (0 to 1) for vertical placement on canvas (bottom to top)
    }));

    return { xLabels, yLabels, minY, maxY };
  };

  const handleHighlightedSectionSelect = (label) => {
    setSelectedHighlightedSections((prev) => {
      const newSelected = [...prev];
      const sectionIndex = newSelected.indexOf(label);

      if (sectionIndex === -1) {
        newSelected.push(label);
      } else {
        newSelected.splice(sectionIndex, 1);
      }

      return newSelected;
    });
  };

  const togglePlayingSection = (idx, start, end) => {
    setPlayingSection(idx);
    setPlayingAudioRange([start, end]);
  };

  useEffect(() => {
    if (
      (playingSection === "waveform" || playingSection === null) &&
      sourceNodeRef.current
    ) {
      sourceNodeRef.current.stop();
      sourceNodeRef.current = null;
    } else if (playingSection !== null && playingSection !== "waveform") {
      if (sourceNodeRef.current) {
        sourceNodeRef.current.stop();
        sourceNodeRef.current = null;
      }

      if (audioContextRef.current && audioBuffer && playingAudioRange) {
        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContextRef.current.destination);
        source.start(
          0,
          playingAudioRange[0] / features.sample_rate,
          (playingAudioRange[1] - playingAudioRange[0]) / features.sample_rate
        );
        source.onended = () => {
          if (sourceNodeRef.current === source) {
            setPlayingSection(null);
            sourceNodeRef.current = null;
          }
        };
        sourceNodeRef.current = source;
      }
    }
  }, [playingSection]);

  useEffect(() => {
    if (audioData && features) {
      const axes = calculateWaveformAxes(audioData, features.sample_rate);
      setWaveformAxes(axes);
      const normVariabilityAxes = calculateVariabilityAxes(
        features.normalized_timbre_variability
      );
      setVariabilityAxes(normVariabilityAxes);

      const sectionLabels = [
        "Timbre",
        "Loudness",
        "Pitch",
        "Staccato",
        "Legato",
      ];
      const sectionColors = ["red", "green", "orange", "pink", "purple"];

      const labeledHighlightedSections = features.variable_sections.reduce(
        (acc, section, idx) => {
          const [start, end] = section;
          const label = sectionLabels[idx];
          const color = sectionColors[idx];
          const existingSection = acc.find(
            (s) => s.start === start && s.end === end
          );

          if (existingSection) {
            existingSection.label += ` + ${label}`;
          } else {
            acc.push({ start, end, label, color });
          }

          return acc;
        },
        []
      );
      setHighlightedSections(labeledHighlightedSections);
      setSelectedHighlightedSections(
        labeledHighlightedSections.map((section) => section.label)
      );
    }
  }, [audioData, features]);

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100">
      <Header title="Audio Analyzer" />

      <div className="flex flex-row space-x-6 h-full xl:w-3/5 lg:w-3/4">
        <div className="flex flex-col justify-between w-1/6"></div>
        {/* Upload File/Playback Audio File */}
        {!file ? (
          <FileUploader
            audioContext={audioContextRef.current}
            setFile={setFile}
            setAudioBuffer={setAudioBuffer}
            setAudioData={setAudioData}
            setFeatures={setFeatures}
          />
        ) : (
          <div className="w-full">
            {audioBuffer && features && waveformAxes && highlightedSections && (
              <WaveformPlayback
                file={file}
                playingSection={playingSection}
                setPlayingSection={setPlayingSection}
                setFile={setFile}
                setAudioBuffer={setAudioBuffer}
                setFeatures={setFeatures}
                highlightedRegions={highlightedSections}
                sampleRate={features.sample_rate}
              />
            )}
          </div>
        )}
        <div className="flex flex-col justify-between w-1/4"></div>
      </div>

      {audioBuffer && features && waveformAxes && highlightedSections && (
        <div className="flex flex-col h-full w-full xl:w-3/5 lg:w-3/4 mt-4 flex-grow py-4 space-y-4">
          {/* First row of graphs */}
          <div className="flex flex-row h-[300px]">
            {/* Visualization Tabs */}
            <Tabs
              activeTab={activeVisualizationTab}
              setActiveTab={setActiveVisualizationTab}
              tabs={visualizationTabs}
            />

            {/* Graph Display */}
            <div className="p-4 bg-blue-50 rounded-r-lg border-2 border-blue-500 border-solid relative w-full">
              {(() => {
                let title, data, highlightedSectionsData, axes;

                if (activeVisualizationTab === "Highlights") {
                  title = "Highlighted Features";
                  data = audioBuffer.getChannelData(0);
                  highlightedSectionsData = highlightedSections.filter(
                    (section) =>
                      selectedHighlightedSections.includes(section.label)
                  );
                  axes = waveformAxes;
                } else if (activeVisualizationTab === "Variability") {
                  title = "Normalized Variability";
                  data = [
                    {
                      data: features.normalized_timbre_variability,
                      lineColor: "red",
                    },
                    {
                      data: features.normalized_loudness_variability,
                      lineColor: "green",
                    },
                    {
                      data: features.normalized_pitch_variability,
                      lineColor: "orange",
                    },
                    {
                      data: features.normalized_articulation_variability,
                      lineColor: "blue",
                    },
                  ];
                  highlightedSectionsData = [];
                  axes = variabilityAxes;
                }

                return (
                  <AudioFeaturesDisplay
                    title={title}
                    data={data}
                    axes={axes}
                    highlightedSections={highlightedSectionsData}
                  />
                );
              })()}
            </div>
            <div className="flex flex-col w-3/12 p-4 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid">
              <div className="mb-12 font-semibold text-center">Legend</div>
              <div className="justify-center">
                {highlightedSections.map((section, idx) => (
                  <div key={idx} className="flex items-center space-x-2">
                    <IconButton
                      onClick={() =>
                        togglePlayingSection(idx, section.start, section.end)
                      }
                      style={{ color: section.color }}
                    >
                      {playingSection === idx ? (
                        <PauseCircleIcon />
                      ) : (
                        <PlayCircleIcon />
                      )}
                    </IconButton>
                    <span className="text-sm text-slate-800">
                      {section.label}
                    </span>
                    <input
                      type="checkbox"
                      className={`h-4 w-4 border rounded border-${section.color}-300 text-${section.color}-500 accent-${section.color}-500 focus:ring-blue-500`}
                      checked={selectedHighlightedSections.includes(
                        section.label
                      )}
                      onChange={() =>
                        handleHighlightedSectionSelect(section.label)
                      }
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Second row of graphs */}
          <div className="flex flex-row h-[300px]">
            {/* Visualization Tabs */}
            <Tabs
              activeTab={activeFeatureTab}
              setActiveTab={setActiveFeatureTab}
              tabs={featureTabs}
            />
            {/* <div className="flex flex-col justify-between w-1/6">
              <button
                className={`flex-grow py-2 px-4 text-lg text-white flex-grow rounded-t-lg ${
                  activeFeatureTab === "Loudness"
                    ? "border-none font-semibold bg-[#008000]"
                    : "transition font-semibold ease-in-out delay-50 border-none bg-[#008000] bg-opacity-50 hover:bg-opacity-75"
                }`}
                onClick={() => setActiveFeatureTab("Loudness")}
              >
                Loudness
              </button>
              <button
                className={`flex-grow py-2 px-4 text-lg text-white flex-grow rounded-b-lg ${
                  activeFeatureTab === "Pitch"
                    ? "border-none font-semibold bg-[#FFA500]"
                    : "transition font-semibold ease-in-out delay-50 border-none bg-[#FFA500] bg-opacity-50 hover:bg-opacity-75"
                }`}
                onClick={() => setActiveFeatureTab("Pitch")}
              >
                Pitch
              </button>
            </div> */}

            {/* Graph Display */}
            <div className="p-4 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid relative w-full">
              {activeFeatureTab === "Loudness" &&
                (() => {
                  const axes = calculateAxes(
                    features.loudness_smoothed,
                    features.sample_rate,
                    features.hop_length
                  );

                  return (
                    <div className="flex flex-col items-center h-full">
                      <div className="text-center font-semibold text-slate-800">
                        Loudness
                      </div>
                      <AudioFeaturesGraph
                        data={features.loudness_smoothed}
                        xLabels={waveformAxes.xLabels}
                        yLabels={axes.yLabels}
                        minY={axes.minY}
                        maxY={axes.maxY}
                        color="green"
                      />
                    </div>
                  );
                })()}
              {activeFeatureTab === "Pitch" &&
                (() => {
                  const axes = calculateAxes(
                    features.pitches_smoothed,
                    features.sample_rate,
                    features.hop_length
                  );

                  return (
                    <div className="flex flex-col items-center h-full">
                      <div className="text-center font-semibold text-slate-800">
                        Pitch
                      </div>
                      <AudioFeaturesGraph
                        data={features.pitches_smoothed}
                        xLabels={waveformAxes.xLabels}
                        yLabels={axes.yLabels}
                        minY={axes.minY}
                        maxY={axes.maxY}
                        color="orange"
                      />
                    </div>
                  );
                })()}
            </div>
            <div className="flex flex-col w-3/12 p-4 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid">
              {" "}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioAnalyzer;
