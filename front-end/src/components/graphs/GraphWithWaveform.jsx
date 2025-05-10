import { useRef, useState } from "react";
import LineGraph from "./LineGraph";
import WaveformPlayer from "./WaveformPlayer";
import { FaPlay, FaPause } from "react-icons/fa";
import IconButton from "../buttons/IconButton";

const width = 800;
const leftMargin = 50;
const rightMargin = 20;
const buttonWidth = 40; // w-10 = 40px
const graphHeight = 400;
const waveformHeight = 50;

const GraphWithWaveform = ({
  audioURL,
  featureData,
  sampleRate,
  highlightedSections,
}) => {
  const wavesurferRef = useRef();
  const [isPlaying, setIsPlaying] = useState(false);

  return (
    <div className="flex flex-col items-center" style={{ width }}>
      <LineGraph
        data={featureData}
        width={width}
        height={graphHeight}
        xLabel="time (s)"
        yLabel="feature"
        highlightedSections={highlightedSections}
      />
      <div className="flex flex-row items-center w-full">
        <div
          style={{
            width: buttonWidth,
            display: "flex",
            justifyContent: "center",
          }}
        >
          <IconButton
            icon={isPlaying ? FaPause : FaPlay}
            onClick={() => {
              if (wavesurferRef.current) {
                wavesurferRef.current.playPause();
              }
              setIsPlaying((prev) => !prev);
            }}
            colorClass="text-darkpink"
            bgClass="bg-transparent"
            sizeClass="w-10 h-10"
            ariaLabel="play audio"
          />
        </div>
        <div
          style={{
            width: width - leftMargin - rightMargin,
            marginLeft: leftMargin - buttonWidth,
            marginRight: rightMargin,
            boxSizing: "border-box",
          }}
        >
          <WaveformPlayer
            key={audioURL}
            audioUrl={audioURL}
            width={width - leftMargin - rightMargin}
            height={waveformHeight}
            highlightedSections={highlightedSections}
            wavesurferRef={wavesurferRef}
            setIsPlaying={setIsPlaying}
          />
        </div>
      </div>
    </div>
  );
};

export default GraphWithWaveform;
