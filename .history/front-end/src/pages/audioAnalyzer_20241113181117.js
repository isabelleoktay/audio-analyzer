import React, { useState, useRef, useCallback, useEffect } from "react";
import Header from "../components/Header";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import WaveSurfer from "wavesurfer.js";
import AudioFeaturesGraph from "../components/AudioFeaturesGraph";
import NormalizedVariabilityChart from "../components/NormalizedVariabilityChart";
import { processAudio } from "../utils/api";

const AudioAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [waveformAxes, setWaveformAxes] = useState(null);
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

  console.log(audioBuffer);

  const getAudioBuffer = useCallback(async (audioFile) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();
    }
    const reader = new FileReader();
    return new Promise((resolve, reject) => {
      reader.onload = (event) => {
        audioContextRef.current.decodeAudioData(
          event.target.result,
          (buffer) => {
            resolve(buffer);
          },
          reject
        );
      };
      reader.readAsArrayBuffer(audioFile);
    });
  }, []);

  const processAudioBuffer = async (file) => {
    try {
      const processAudioResult = await processAudio(file);
      setFeatures(processAudioResult);

      console.log("PYTHON RESULT");
      console.log(processAudioResult);
    } catch (error) {
      console.error("Error processing audio:", error);
    }
  };

  const handleFileUpload = async (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile && uploadedFile.type.startsWith("audio/")) {
      setFile(uploadedFile);
      const buffer = await getAudioBuffer(uploadedFile);
      setAudioBuffer(buffer);
      setAudioData(buffer.getChannelData(0));
      await processAudioBuffer(uploadedFile);
    } else {
      alert("Please upload an audio file.");
    }
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    const uploadedFile = event.dataTransfer.files[0];
    if (uploadedFile && uploadedFile.type.startsWith("audio/")) {
      setFile(uploadedFile);
      const buffer = await getAudioBuffer(uploadedFile);
      setAudioBuffer(buffer);
      await processAudioBuffer(uploadedFile);
    } else {
      alert("Please upload an audio file.");
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const removeFile = () => {
    setFile(null);
    setPlayingSection(null);
    setAudioBuffer(null);
    setFeatures(null);
    if (waveSurferRef.current) {
      waveSurferRef.current.destroy();
      waveSurferRef.current = null;
    }
  };

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

  const calculateWaveformAxes = (data, sampleRate) => {
    // Calculate min and max values for the y-axis
    const minY = data.reduce((min, val) => Math.min(min, val), Infinity);
    const maxY = data.reduce((max, val) => Math.max(max, val), -Infinity);

    // Calculate total duration in seconds (number of samples / sample rate)
    const duration = audioBuffer.duration;
    console.log("waveform duration: " + duration);

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

  const toggleWaveform = () => {
    if (waveSurferRef.current) {
      if (waveSurferRef.current.isPlaying()) {
        waveSurferRef.current.pause();
        if (playingSection === "waveform") {
          setPlayingSection(null);
        }
      } else {
        waveSurferRef.current.play();
      }
    }
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
    if (file && waveformContainerRef.current) {
      if (waveSurferRef.current) {
        waveSurferRef.current.destroy();
      }

      waveSurferRef.current = WaveSurfer.create({
        container: waveformContainerRef.current,
        waveColor: "#60a5fa",
        progressColor: "#3b82f6",
        barWidth: 2,
        cursorColor: "#3b82f6",
        height: 80,
        responsive: true,
      });

      waveSurferRef.current.load(URL.createObjectURL(file));

      waveSurferRef.current.on("play", () => {
        setPlayingSection("waveform");
      });

      waveSurferRef.current.on("finish", () => {
        setPlayingSection(null);
      });
    }

    return () => {
      if (waveSurferRef.current) {
        waveSurferRef.current.destroy();
      }
    };
  }, [file]);

  useEffect(() => {
    if (playingSection !== null && playingSection !== "waveform") {
      waveSurferRef.current.pause();
    }
  }, [playingSection]);

  useEffect(() => {
    if (audioData && features) {
      const axes = calculateWaveformAxes(audioData, features.sample_rate);
      setWaveformAxes(axes);

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
      {/* Header */}
      <header className="w-full bg-blue-500 py-3 text-center font-chakra">
        <h1 className="text-5xl text-white font-semibold tracking-widest">
          Audio Analyzer
        </h1>
      </header>

      {/* File Upload Area */}
      {!file ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="mt-8 p-8 border-2 border-dashed border-blue-300 bg-blue-100 text-center rounded-lg w-3/4 max-w-lg cursor-default"
        >
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
            id="fileInput"
          />
          <label htmlFor="fileInput" className="block text-gray-600">
            <div className="flex flex-col items-center">
              <UploadFileIcon className="text-gray-600 sm:text-2xl md:text-2xl lg:text-3xl xl:text-3xl mb-1" />
              <p>
                Drag and drop file here or{" "}
                <span className="text-blue-500 underline font-bold cursor-pointer">
                  upload file
                </span>
              </p>
            </div>
          </label>
        </div>
      ) : (
        <div className="flex flex-col mt-8 xl:w-3/5 lg:w-3/4 p-4 bg-blue-100 rounded-lg border-2 border-blue-300 border-solid">
          <div className="flex items-center mb-1">
            <div className="text-sm font-semibold text-blue-500">
              {file.name}
            </div>
            {/* Remove File Button */}
            <button
              onClick={removeFile}
              className={`ml-auto hover:cursor-pointer transition ease-in-out delay-50 text-xs text-center text-white hover:text-white text-sm border-transparent focus:border-transparent focus:ring-0 focus:outline-none bg-red-500 hover:bg-red-400 py-1 px-2 rounded-l-none outline-none rounded`}
            >
              Remove File
            </button>
          </div>
          <div className="flex items-center">
            {/* Play/Pause Button */}
            <IconButton
              aria-label={playingSection === "waveform" ? "pause" : "play"}
              onClick={toggleWaveform}
              className="mr-1"
            >
              {playingSection === "waveform" ? (
                <PauseCircleIcon className="text-blue-500 text-3xl" />
              ) : (
                <PlayCircleIcon className="text-blue-500 text-3xl" />
              )}
            </IconButton>

            {/* Waveform Container */}
            <div className="flex-grow" ref={waveformContainerRef}></div>
          </div>
        </div>
      )}
      {audioBuffer && features && waveformAxes && highlightedSections && (
        <div className="flex flex-col h-full w-full xl:w-3/5 lg:w-3/4 mt-4 flex-grow py-4 space-y-4">
          {/* First row of graphs */}
          <div className="flex flex-row space-x-6 h-[300px]">
            {/* Visualization Tabs */}
            <div className="flex flex-col justify-between w-1/6">
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

            {/* Graph Display */}
            <div className="p-4 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid relative w-full">
              {activeVisualizationTab === "Highlights" && (
                <div className="flex flex-col items-center h-full">
                  <div className="text-center font-semibold text-slate-800">
                    Highlighted Features
                  </div>
                  <AudioFeaturesGraph
                    data={audioBuffer.getChannelData(0)}
                    xLabels={waveformAxes.xLabels}
                    yLabels={waveformAxes.yLabels}
                    minY={waveformAxes.minY}
                    maxY={waveformAxes.maxY}
                    highlightedSections={highlightedSections.filter((section) =>
                      selectedHighlightedSections.includes(section.label)
                    )}
                  />
                </div>
              )}
              {activeVisualizationTab === "Variability" && (
                <NormalizedVariabilityChart
                  timbre={features.normalized_timbre_variability}
                  loudness={features.normalized_loudness_variability}
                  pitch={features.normalized_pitch_variability}
                  articulation={features.normalized_articulation_variability}
                  xTicks={features.normalized_time_axis}
                />
              )}
            </div>
            <div className="flex flex-col w-3/12 p-4 bg-blue-50 rounded-lg border-2 border-blue-300 border-solid">
              <div className="mb-12 font-semibold text-center">Audio</div>
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
          <div className="flex flex-row space-x-6 h-[300px]">
            {/* Visualization Tabs */}
            <div className="flex flex-col justify-between w-1/6">
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
            </div>

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
