import React, { useState, useRef, useEffect } from "react";
import Header from "../components/Header";
import FileUploader from "../components/FileUploader";
import WaveformPlayback from "../components/WaveformPlayback";
import AudioFeaturesDisplay from "../components/AudioFeaturesDisplay";
import Tabs from "../components/Tabs";
import CollapsibleLegend from "../components/CollapsibleLegend";

const visualizationTabs = [
  { name: "Highlights", color: "blue-500" },
  { name: "Variability", color: "blue-500" },
];

const featureTabs = [
  { name: "Loudness", color: "blue-500" },
  { name: "Pitch", color: "blue-500" },
  { name: "Tempo", color: "blue-500" },
];

const sectionLabels = ["Timbre", "Loudness", "Pitch", "Staccato", "Legato"];
const sectionColors = ["red", "green", "orange", "pink", "purple"];

const AudioAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [file2, setFile2] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [waveformAxes, setWaveformAxes] = useState(null);
  const [variabilityAxes, setVariabilityAxes] = useState(null);
  const [loudnessAxes, setLoudnessAxes] = useState(null);
  const [pitchAxes, setPitchAxes] = useState(null);
  const [tempoAxes, setTempoAxes] = useState(null);
  const [highlightedSections, setHighlightedSections] = useState(null);
  const [normalizedVariabilities, setNormalizedVariabilities] = useState(null);
  const [selectedHighlightedSections, setSelectedHighlightedSections] =
    useState([]);
  const [playingSection, setPlayingSection] = useState(null);
  const [features, setFeatures] = useState(null);
  const [activeFeatureTab, setActiveFeatureTab] = useState("Loudness");
  const [activeVisualizationTab, setActiveVisualizationTab] =
    useState("Highlights");
  // const [playingAudioRange, setPlayingAudioRange] = useState(null);
  const [minNote, setMinNote] = useState("");
  const [maxNote, setMaxNote] = useState("");
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const playingAudioRangeRef = useRef(null);

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
    playingAudioRangeRef.current = [start, end];
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

      if (
        audioContextRef.current &&
        audioBuffer &&
        playingAudioRangeRef.current &&
        features.sample_rate
      ) {
        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContextRef.current.destination);
        source.start(
          0,
          playingAudioRangeRef.current[0] / features.sample_rate,
          (playingAudioRangeRef.current[1] - playingAudioRangeRef.current[0]) /
            features.sample_rate
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
  }, [playingSection, audioBuffer, features?.sample_rate]);

  useEffect(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();
    }
  }, []);

  useEffect(() => {
    const calculateAxes = (data, type, yMargin = 0) => {
      let minY, maxY, duration;
      duration = audioBuffer.duration;

      if (type === "feature") {
        minY = Math.min(...data);
        maxY = Math.max(...data);
      } else {
        minY = data.reduce((min, val) => Math.min(min, val), Infinity);
        maxY = data.reduce((max, val) => Math.max(max, val), -Infinity);
      }

      if (yMargin !== 0) {
        minY = Math.max(0, minY - yMargin);
        maxY = maxY + yMargin;
      }

      // X-axis time labels based on sample rate and hop length
      const xLabels = Array.from({ length: 6 }, (_, i) => ({
        label: `${((duration * i) / 5).toFixed(2)}s`,
        position: i / 5, // normalized position (0 to 1) to use within width
      }));

      // Y-axis labels based on min and max values
      let yLabels;
      if (type === "feature" && data === features.pitches_smoothed) {
        // Convert frequency to note names
        const freqToNote = (freq) => {
          if (freq <= 0) return "N/A"; // Handle invalid frequencies
          const midiNote = Math.round(69 + 12 * Math.log2(freq / 440)); // Convert freq to MIDI
          const noteName =
            midiNote >= 0 && midiNote <= 127
              ? [
                  "C",
                  "C#",
                  "D",
                  "D#",
                  "E",
                  "F",
                  "F#",
                  "G",
                  "G#",
                  "A",
                  "A#",
                  "B",
                ][midiNote % 12]
              : "";
          const octave = Math.floor(midiNote / 12) - 1;
          return noteName && octave !== undefined
            ? `${noteName}${octave}`
            : "N/A";
        };

        yLabels = Array.from({ length: 10 }, (_, i) => {
          const freq = minY + ((maxY - minY) * i) / 9;
          return {
            label: freqToNote(freq),
            position: i / 9, // normalized (1 to 0) for top to bottom
          };
        });
      } else {
        // Standard numerical Y-axis labels
        yLabels = Array.from({ length: 5 }, (_, i) => ({
          label: (minY + ((maxY - minY) * i) / 4).toFixed(2),
          position: i / 4, // normalized (1 to 0) for top to bottom
        }));
      }
      return { xLabels, yLabels, minY, maxY };
    };

    if (audioData && features) {
      console.log("FEATURES");
      console.log(features);
      setWaveformAxes(calculateAxes(audioData));
      setLoudnessAxes(calculateAxes(features.loudness_smoothed, "feature"));
      setPitchAxes(calculateAxes(features.pitches_smoothed, "feature"));
      setTempoAxes(calculateAxes(features.dynamic_tempo, "feature", 50));
      setVariabilityAxes(calculateAxes(features.normalized_timbre_variability));

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
  }, [audioData, features, audioBuffer?.duration]);

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100">
      <Header title="Audio Analyzer" />

      <div className="flex flex-row space-x-6 h-full xl:w-3/5 lg:w-1/2">
        {/* Upload File/Playback Audio File */}
        {!(audioBuffer && features && waveformAxes && highlightedSections) ? (
          <FileUploader
            audioContextRef={audioContextRef}
            setFile={setFile}
            file={file}
            setFile2={setFile2}
            file2={file2}
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
          <div className="flex flex-row h-[300px] shadow-md">
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
                  data = features.audio;
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
          <div className="flex flex-row h-[300px] shadow-md">
            {/* Visualization Tabs */}
            <Tabs
              activeTab={activeFeatureTab}
              setActiveTab={setActiveFeatureTab}
              tabs={featureTabs}
            />

            {/* Graph Display */}
            <div
              className={`p-4 bg-blue-50 ${
                activeFeatureTab === "Tempo"
                  ? "border-y-2 border-l-2"
                  : "rounded-r-lg border-2"
              } border-blue-500 border-solid z-30 relative w-full`}
            >
              {(() => {
                let title, data, axes, color;
                const highlightedSectionsData = [];

                if (activeFeatureTab === "Loudness") {
                  title = "Loudness";
                  data = features.loudness_smoothed;
                  axes = loudnessAxes;
                  color = "green";
                } else if (activeFeatureTab === "Pitch") {
                  title = "Pitch";
                  data = features.pitches_smoothed;
                  axes = pitchAxes;
                  color = "orange";
                } else if (activeFeatureTab === "Tempo") {
                  title = "Tempo";
                  data = [
                    {
                      data: features.dynamic_tempo,
                      lineColor: "deepskyblue",
                      label: "Dynamic Tempo",
                    },
                    {
                      data: features.global_tempo,
                      lineColor: "navy",
                      label: "Global Tempo:",
                      dashed: true,
                    },
                  ];
                  axes = tempoAxes;
                  color = "";
                }

                return (
                  <AudioFeaturesDisplay
                    title={title}
                    data={data}
                    axes={axes}
                    highlightedSections={highlightedSectionsData}
                    color={color}
                  />
                );
              })()}
            </div>
            {activeFeatureTab === "Tempo" && (
              <CollapsibleLegend
                sections={[
                  {
                    data: features.dynamic_tempo,
                    lineColor: "deepskyblue",
                    label: "Dynamic Tempo",
                  },
                  {
                    data: features.global_tempo,
                    lineColor: "navy",
                    label: "Global Tempo",
                    subLabel: `(${parseInt(features.global_tempo[0])} BPM)`,
                    dashed: true,
                  },
                ]}
                selectedSections={selectedHighlightedSections}
                activeTab={activeFeatureTab}
                startOpen={true}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioAnalyzer;
