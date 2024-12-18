import React, { useState, useRef, useEffect } from "react";
import Header from "../components/Header";
import FileUploader from "../components/FileUploader";
import WaveformPlayback from "../components/WaveformPlayback";
import AudioFeaturesDisplay from "../components/AudioFeaturesDisplay";
import Tabs from "../components/Tabs";
import CollapsibleLegend from "../components/CollapsibleLegend";
import AudioFeaturesGraph from "../components/AudioFeaturesGraph";

const visualizationTabs = [
  { name: "Highlights", color: "blue-500" },
  { name: "Variability", color: "blue-500" },
];

const featureTabs = [
  { name: "Loudness", color: "blue-500" },
  { name: "Pitch", color: "blue-500" },
];

const AudioAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [waveformAxes, setWaveformAxes] = useState(null);
  const [variabilityAxes, setVariabilityAxes] = useState(null);
  const [highlightedSections, setHighlightedSections] = useState(null);
  const [normalizedVariabilities, setNormalizedVariabilities] = useState(null);
  const [selectedHighlightedSections, setSelectedHighlightedSections] =
    useState([]);
  const [playingSection, setPlayingSection] = useState(null);
  const [features, setFeatures] = useState(null);
  const [activeFeatureTab, setActiveFeatureTab] = useState("Loudness");
  const [activeVisualizationTab, setActiveVisualizationTab] =
    useState("Highlights");
  const [playingAudioRange, setPlayingAudioRange] = useState(null);
  const [minNote, setMinNote] = useState("");
  const [maxNote, setMaxNote] = useState("");
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);

  const calculateAxes = (data, sampleRate, hopLength, type) => {
    let minY, maxY, duration;

    if (type === "feature") {
      minY = Math.min(...data);
      maxY = Math.max(...data);
      duration = (data.length * hopLength) / sampleRate;
    } else {
      minY = data.reduce((min, val) => Math.min(min, val), Infinity);
      maxY = data.reduce((max, val) => Math.max(max, val), -Infinity);
      duration = audioBuffer.duration;
    }

    // X-axis time labels based on sample rate and hop length
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
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();
    }
  }, []);

  useEffect(() => {
    if (audioData && features) {
      const axes = calculateAxes(audioData, features.sample_rate);
      setWaveformAxes(axes);
      const normVariabilityAxes = calculateAxes(
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
      setNormalizedVariabilities([
        {
          data: features.normalized_timbre_variability,
          lineColor: "red",
          label: "Timbre",
        },
        {
          data: features.normalized_loudness_variability,
          lineColor: "green",
          label: "Loudness",
        },
        {
          data: features.normalized_pitch_variability,
          lineColor: "orange",
          label: "Pitch",
        },
        {
          data: features.normalized_articulation_variability,
          lineColor: "purple",
          label: "Articulation",
        },
      ]);
    }
  }, [audioData, features]);

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100">
      <Header title="Audio Analyzer" />

      <div className="flex flex-row space-x-6 h-full xl:w-3/5 lg:w-1/2">
        {/* Upload File/Playback Audio File */}
        {!(audioBuffer && features && waveformAxes && highlightedSections) ? (
          <FileUploader
            audioContext={audioContextRef.current}
            setFile={setFile}
            file={file}
            setAudioBuffer={setAudioBuffer}
            setAudioData={setAudioData}
            setFeatures={setFeatures}
            minNote={minNote}
            maxNote={maxNote}
            setMinNote={setMinNote}
            setMaxNote={setMaxNote}
          />
        ) : (
          <div className="w-full">
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
          </div>
        )}
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
            <div className="p-4 bg-blue-50 border-y-2 border-l-2 border-blue-500 z-30 border-solid relative w-full">
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
                  data = normalizedVariabilities;
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
            <CollapsibleLegend
              sections={
                activeVisualizationTab === "Highlights"
                  ? highlightedSections
                  : normalizedVariabilities
              }
              selectedSections={selectedHighlightedSections}
              togglePlayingSection={togglePlayingSection}
              handleSectionSelect={handleHighlightedSectionSelect}
              playingSection={playingSection}
              activeTab={activeVisualizationTab}
            />
          </div>

          {/* Second row of graphs */}
          <div className="flex flex-row h-[300px]">
            {/* Visualization Tabs */}
            <Tabs
              activeTab={activeFeatureTab}
              setActiveTab={setActiveFeatureTab}
              tabs={featureTabs}
            />

            {/* Graph Display */}
            <div className="p-4 bg-blue-50 rounded-r-lg border-2 border-blue-500 border-solid z-30 relative w-full">
              {activeFeatureTab === "Loudness" &&
                (() => {
                  const axes = calculateAxes(
                    features.loudness_smoothed,
                    features.sample_rate,
                    features.hop_length,
                    "feature"
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
                    features.hop_length,
                    "feature"
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
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioAnalyzer;
