import React, { useState, useRef, useEffect } from "react";
import Header from "../components/Header";
import FileUploader from "../components/FileUploader";
import WaveformPlayback from "../components/WaveformPlayback";
import AudioFeaturesDisplay from "../components/AudioFeaturesDisplay";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import AudioFeaturesGraph from "../components/AudioFeaturesGraph";
import NormalizedVariabilityChart from "../components/NormalizedVariabilityChart";

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
  const waveSurferRef = useRef(null);
  const waveformContainerRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);

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
    if (playingSection !== null && playingSection !== "waveform") {
      waveSurferRef.current.pause();
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

      <div className="flex flex-row w-full xl:w-3/5 lg:w-3/4">
        {/* Left Column */}
        <div className="flex flex-col justify-between w-1/6 flex-1">
          <button
            className={`flex-grow py-2 px-4 text-lg text-white w-full font-semibold border-none bg-blue-500 rounded-t-lg ${
              activeVisualizationTab === "Highlights"
                ? ""
                : "transition ease-in-out delay-50 bg-opacity-50 hover:bg-opacity-75"
            }`}
            onClick={() => setActiveVisualizationTab("Highlights")}
          >
            Highlights
          </button>
          <button
            className={`flex-grow py-2 px-4 text-lg text-white w-full font-semibold border-none bg-blue-500 rounded-b-lg ${
              activeVisualizationTab === "Variability"
                ? ""
                : "transition ease-in-out delay-50 bg-opacity-50 hover:bg-opacity-75"
            }`}
            onClick={() => setActiveVisualizationTab("Variability")}
          >
            Variability
          </button>
        </div>

        {/* Main Content */}
        <div className="flex flex-col items-center w-full h-full flex-2">
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
            <WaveformPlayback
              file={file}
              playingSection={playingSection}
              setPlayingSection={setPlayingSection}
              setFile={setFile}
              setAudioBuffer={setAudioBuffer}
              setFeatures={setFeatures}
            />
          )}
          {audioBuffer &&
            features &&
            waveformAxes &&
            highlightedSections &&
            (() => {
              let title, data, highlightedSectionsData, axes;

              if (activeVisualizationTab === "Highlights") {
                title = "Highlighted Features";
                data = [
                  { data: audioBuffer.getChannelData(0), lineColor: "black" },
                ];
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
                <div className="p-4 mt-8 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid relative w-full h-[300px]">
                  <AudioFeaturesDisplay
                    title={title}
                    data={data}
                    axes={axes}
                    highlightedSections={highlightedSectionsData}
                  />
                </div>
              );
            })()}
          {/* {audioBuffer && features && waveformAxes && highlightedSections && (
            <div className="p-4 mt-8 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid relative w-full h-[300px]">
              {activeVisualizationTab === "Highlights" && (
                <AudioFeaturesDisplay
                  title="Highlighted Features"
                  data={[
                    { data: audioBuffer.getChannelData(0), lineColor: "black" },
                  ]}
                  axes={waveformAxes}
                  highlightedSections={highlightedSections.filter((section) =>
                    selectedHighlightedSections.includes(section.label)
                  )}
                />
              )}
              {activeVisualizationTab === "Variability" && (
                <AudioFeaturesDisplay
                  title="Normalized Variability"
                  data={[
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
                  ]}
                  axes={variabilityAxes}
                />
              )}
            </div>
          )} */}
        </div>

        {/* Right Column */}
        <div className="flex-1"></div>
      </div>
    </div>
  );
};

export default AudioAnalyzer;
