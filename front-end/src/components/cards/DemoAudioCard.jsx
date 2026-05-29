import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.esm.js";
import { presetAudios } from "../../config/presetAudios.js";

const SCROLLING_WAVEFORM = true;
const CONTINUOUS_WAVEFORM = false;

const pulsingRecordStyle = {
  animation: "pulse 2s infinite ease-in-out",
};

const DemoAudioCard = ({
  label,
  onAudioSourceChange,
  onAudioDataChange,
  filterVoiceCategory = null, // "female" | "male" | null (no filter)
}) => {
  const [selectedAudioSource, setSelectedAudioSource] = useState("presets");
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingPreset, setPlayingPreset] = useState(null);
  const [isRecordingMode, setIsRecordingMode] = useState(false);
  const [recordingName, setRecordingName] = useState("untitled");
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordingPlayback, setRecordingPlayback] = useState(false);

  const waveSurferRef = useRef(null);
  const recordRef = useRef(null);

  const VOICE_CATEGORY_MAP = {
    soprano: "female",
    mezzo: "female",
    alto: "female",
    tenor: "male",
    bass: "male",
  };

  // Filter presets by voice category if a filter is active
  const filteredPresets = filterVoiceCategory
    ? presetAudios.filter(
        (p) => VOICE_CATEGORY_MAP[p.voiceType] === filterVoiceCategory,
      )
    : presetAudios;

  // If selected preset no longer matches filter, clear it
  useEffect(() => {
    if (
      selectedPreset &&
      filterVoiceCategory &&
      VOICE_CATEGORY_MAP[selectedPreset.voiceType] !== filterVoiceCategory
    ) {
      setSelectedPreset(null);
      onAudioDataChange?.({
        source: "presets",
        file: null,
        blob: null,
        url: null,
        voiceType: null,
      });
    }
  }, [filterVoiceCategory]);

  const handleSelectAudioSource = (source) => {
    setSelectedAudioSource(source);
    onAudioSourceChange?.(source);

    if (source === "presets" && selectedPreset) {
      onAudioDataChange?.({
        source: "presets",
        file: null,
        blob: null,
        url: selectedPreset.path,
        name: selectedPreset.name,
        presetId: selectedPreset.id,
        voiceType: selectedPreset.voiceType,
      });
    } else if (source === "record" && audioBlob) {
      onAudioDataChange?.({
        source: "record",
        file: null,
        blob: audioBlob,
        url: URL.createObjectURL(audioBlob),
        name: recordingName,
        voiceType: null,
      });
    } else {
      onAudioDataChange?.({
        source: source,
        file: null,
        blob: null,
        url: null,
        voiceType: null,
      });
    }
  };

  useEffect(() => {
    if (
      selectedAudioSource === "presets" &&
      playingPreset &&
      !waveSurferRef.current
    ) {
      const waveSurfer = WaveSurfer.create({
        container: "#demo-waveform",
        waveColor: "rgb(255, 214, 232)",
        progressColor: "rgb(255, 137, 187)",
        interact: true,
        height: 60,
      });

      waveSurferRef.current = waveSurfer;

      waveSurfer.on("finish", () => setIsPlaying(false));

      return () => {
        waveSurfer.destroy();
        waveSurferRef.current = null;
      };
    }
  }, [playingPreset, selectedAudioSource]);

  useEffect(() => {
    if (selectedAudioSource === "record" && isRecordingMode) {
      const waveSurfer = WaveSurfer.create({
        container: "#recording-waveform",
        waveColor: "rgb(255, 214, 232)",
        progressColor: "rgb(255, 137, 187)",
        interact: true,
        height: 100,
      });

      const recordPlugin = RecordPlugin.create({
        renderRecordedAudio: false,
        scrollingWaveform: SCROLLING_WAVEFORM,
        continuousWaveform: CONTINUOUS_WAVEFORM,
      });

      waveSurferRef.current = waveSurfer;
      recordRef.current = waveSurfer.registerPlugin(recordPlugin);

      if (audioBlob) {
        waveSurfer.loadBlob(audioBlob).then(() => {
          waveSurfer.seekTo(0);
          waveSurfer.toggleInteraction(true);
        });
      }

      recordPlugin.on("record-end", (blob) => {
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setRecordingPlayback(false);

        if (selectedAudioSource === "record") {
          onAudioDataChange?.({
            source: "record",
            file: null,
            blob: blob,
            url: url,
            name: recordingName,
            voiceType: null,
          });
        }

        waveSurfer.loadBlob(blob).then(() => {
          waveSurfer.seekTo(0);
          waveSurfer.toggleInteraction(true);
        });
      });

      waveSurfer.on("finish", () => setRecordingPlayback(false));

      return () => waveSurfer.destroy();
    }
  }, [
    selectedAudioSource,
    isRecordingMode,
    audioBlob,
    recordingName,
    onAudioDataChange,
  ]);

  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `
      @keyframes pulse {
        0%, 100% { color: rgb(255, 137, 187); }
        50% { color: #ff1493; }
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const handleSelectPreset = async (preset) => {
    try {
      setSelectedPreset(preset);

      const resp = await fetch(preset.path);
      if (!resp.ok) throw new Error("Failed to fetch preset audio");
      const blob = await resp.blob();
      const ext = preset.path.split(".").pop().split("?")[0] || "wav";
      const filename = `${preset.id}.${ext}`;
      const file = new File([blob], filename, {
        type: blob.type || "audio/wav",
      });

      onAudioSourceChange?.("upload");
      onAudioDataChange?.({
        source: "upload",
        file,
        blob: null,
        url: preset.path,
        name: preset.name,
        presetId: preset.id,
        voiceType: preset.voiceType,
      });
    } catch (err) {
      console.error("Error selecting preset audio:", err);
      onAudioSourceChange?.("presets");
      onAudioDataChange?.({
        source: "presets",
        file: null,
        blob: null,
        url: preset.path,
        name: preset.name,
        presetId: preset.id,
        voiceType: preset.voiceType,
      });
    }
  };

  const handlePlayPreset = (e, preset) => {
    e.stopPropagation();

    if (playingPreset?.id !== preset.id) {
      setPlayingPreset(preset);
      if (waveSurferRef.current) {
        waveSurferRef.current.load(preset.path);
        setTimeout(() => {
          waveSurferRef.current?.play();
          setIsPlaying(true);
        }, 100);
      }
    } else {
      if (isPlaying) {
        waveSurferRef.current?.pause();
        setIsPlaying(false);
      } else {
        waveSurferRef.current?.play();
        setIsPlaying(true);
      }
    }
  };

  const handleRecordClick = () => setIsRecordingMode(true);
  const handleCancelRecord = () => setIsRecordingMode(false);

  const handleRecordButtonClick = async () => {
    if (isRecording) {
      await handleStopRecording();
    } else if (audioBlob) {
      handleResetRecording();
      setTimeout(() => handleStartRecording(), 100);
    } else {
      handleStartRecording();
    }
  };

  const handleStartRecording = async () => {
    await recordRef.current.startRecording();
    setIsRecording(true);
  };

  const handleStopRecording = async () => {
    await recordRef.current.stopRecording();
    setIsRecording(false);
  };

  const handleResetRecording = () => {
    setAudioBlob(null);
    waveSurferRef.current.empty();

    if (selectedAudioSource === "record") {
      onAudioDataChange?.({
        source: "record",
        file: null,
        blob: null,
        url: null,
        voiceType: null,
      });
    }
  };

  const handlePlayPauseRecording = () => {
    if (recordingPlayback) {
      waveSurferRef.current.pause();
      setRecordingPlayback(false);
    } else {
      waveSurferRef.current.play();
      setRecordingPlayback(true);
    }
  };

  return (
    <div className="w-full flex flex-col gap-1">
      <div className="text-2xl font-medium text-lightpink tracking-wide">
        {label}
      </div>
      <div className="w-full bg-lightgray/15 rounded-3xl p-4 flex flex-col gap-4">
        <div className="flex gap-2">
          <button
            onClick={() => handleSelectAudioSource("presets")}
            className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all duration-200 ${
              selectedAudioSource === "presets"
                ? "bg-lightpink text-blueblack"
                : "bg-lightgray/10 text-lightgray hover:bg-lightgray/15"
            }`}
          >
            Presets
          </button>
          <button
            onClick={() => handleSelectAudioSource("record")}
            className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all duration-200 ${
              selectedAudioSource === "record"
                ? "bg-lightpink text-blueblack"
                : "bg-lightgray/10 text-lightgray hover:bg-lightgray/15"
            }`}
          >
            Record
          </button>
        </div>

        {selectedAudioSource === "presets" ? (
          <>
            {playingPreset && (
              <div className="w-full">
                <div className="text-sm text-lightgray mb-2">
                  Now playing: {playingPreset.name}
                </div>
                <div id="demo-waveform" className="w-full"></div>
              </div>
            )}

            {/* Filter hint */}
            {filterVoiceCategory && (
              <div className="text-xs text-lightpink/70 italic">
                Showing {filterVoiceCategory} voice presets only
              </div>
            )}

            <div className="flex flex-col gap-2 max-h-64 overflow-y-auto">
              {filteredPresets.length === 0 ? (
                <div className="text-sm text-lightgray/50 text-center py-4">
                  No presets available for this voice category yet
                </div>
              ) : (
                filteredPresets.map((preset) => (
                  <div
                    key={preset.id}
                    onClick={() => handleSelectPreset(preset)}
                    className={`flex items-center gap-3 p-3 rounded-2xl cursor-pointer transition-all duration-200 ${
                      selectedPreset?.id === preset.id
                        ? "bg-lightpink/30 border border-lightpink"
                        : "bg-lightgray/10 hover:bg-lightgray/15 border border-transparent"
                    }`}
                  >
                    <button
                      onClick={(e) => handlePlayPreset(e, preset)}
                      className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-lightpink hover:bg-lightpink/80 transition-colors text-blueblack font-bold"
                    >
                      {playingPreset?.id === preset.id && isPlaying ? "⏸" : "▶"}
                    </button>
                    <div className="flex-grow">
                      <div className="text-lightgray font-medium">
                        {preset.name}
                      </div>
                    </div>
                    {selectedPreset?.id === preset.id && (
                      <div className="flex-shrink-0 text-lightpink text-lg">
                        ✓
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>

            {selectedPreset && (
              <div className="text-sm text-lightpink text-center pt-2 border-t border-lightgray/20">
                Selected: {selectedPreset.name}
              </div>
            )}
          </>
        ) : (
          <>
            {!isRecordingMode ? (
              <div className="w-full h-32 flex flex-col items-center justify-center text-lightgray gap-2">
                <div className="text-sm">Record your audio</div>
                <button
                  onClick={handleRecordClick}
                  className="px-6 py-2 bg-lightpink/20 hover:bg-lightpink/30 border border-lightpink rounded-xl text-lightpink font-medium transition-all duration-200"
                >
                  Start Recording
                </button>
              </div>
            ) : (
              <>
                <div id="recording-waveform" className="w-full"></div>
                <div className="flex flex-col gap-3">
                  <input
                    type="text"
                    value={recordingName}
                    onChange={(e) => setRecordingName(e.target.value)}
                    placeholder="Recording name"
                    className="w-full px-4 py-2 bg-lightgray/10 border border-lightgray/20 rounded-xl text-lightgray placeholder-lightgray/50 focus:outline-none focus:border-lightpink transition-colors"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={handleRecordButtonClick}
                      className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all duration-200 ${
                        isRecording
                          ? "bg-red-500/30 text-red-400 hover:bg-red-500/40 border border-red-500/50"
                          : audioBlob
                            ? "bg-lightpink text-blueblack hover:bg-lightpink/80"
                            : "bg-lightpink/20 text-lightpink hover:bg-lightpink/30 border border-lightpink"
                      }`}
                      style={isRecording ? pulsingRecordStyle : {}}
                    >
                      {isRecording
                        ? "● Stop"
                        : audioBlob
                          ? "Re-record"
                          : "● Record"}
                    </button>
                    <button
                      onClick={handleCancelRecord}
                      className="flex-1 py-2 px-4 rounded-xl font-medium bg-lightgray/10 text-lightgray hover:bg-lightgray/15 transition-all duration-200"
                    >
                      Cancel
                    </button>
                  </div>
                  {audioBlob && (
                    <div className="flex gap-2">
                      <button
                        onClick={handlePlayPauseRecording}
                        className="flex-1 py-2 px-4 rounded-xl font-medium bg-lightpink/20 text-lightpink hover:bg-lightpink/30 border border-lightpink transition-all duration-200"
                      >
                        {recordingPlayback ? "⏸ Pause" : "▶ Play"}
                      </button>
                      <button
                        onClick={handleResetRecording}
                        className="flex-1 py-2 px-4 rounded-xl font-medium bg-lightgray/10 text-lightgray hover:bg-lightgray/15 transition-all duration-200"
                      >
                        Clear
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default DemoAudioCard;
