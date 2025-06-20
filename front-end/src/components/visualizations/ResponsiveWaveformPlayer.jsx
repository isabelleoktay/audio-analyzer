import { useEffect, useRef, useMemo, useCallback } from "react";
import IconButton from "../buttons/IconButton";
import { FaPlay, FaPause } from "react-icons/fa";
import { useWavesurfer } from "@wavesurfer/react";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions";
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline";

const ResponsiveWaveformPlayer = ({
  audioUrl,
  highlightedSections,
  waveColor = "#FFD6E8",
  progressColor = "#FF89BB",
}) => {
  const containerRef = useRef();
  const timelineRef = useRef();
  const regionsPlugin = useMemo(() => RegionsPlugin.create(), []);
  const timelinePlugin = useMemo(
    () =>
      TimelinePlugin.create({
        container: timelineRef.current, // Attach the timeline to this container
      }),
    []
  );
  const plugins = useMemo(
    () => [regionsPlugin, timelinePlugin],
    [regionsPlugin, timelinePlugin]
  );

  const { wavesurfer, isReady, isPlaying } = useWavesurfer({
    container: containerRef,
    responsive: true,
    normalize: true,
    barWidth: 2,
    height: 50,
    waveColor: waveColor,
    progressColor: progressColor,
    url: audioUrl,
    plugins: plugins,
  });

  const handlePlayPause = useCallback(() => {
    wavesurfer && wavesurfer.playPause();
  }, [wavesurfer]);

  useEffect(() => {
    if (isReady) {
      regionsPlugin.clearRegions();

      highlightedSections.forEach(({ start, end }) => {
        regionsPlugin.addRegion({
          color: "rgba(255, 203, 107, 0.25)",
          start: start,
          end: end,
          drag: false,
          resize: false,
        });
      });

      regionsPlugin.on("region-clicked", (region, e) => {
        region.setOptions({ color: "rgba(255, 203, 107, 0.5)" });
        e.stopPropagation(); // prevent triggering a click on the waveform
        region.play(true);
      });

      regionsPlugin.on("region-out", (region) => {
        region.setOptions({ color: "rgba(255, 203, 107, 0.25)" });
      });
    }

    return () => {
      if (wavesurfer) {
        regionsPlugin.unAll();
      }
    };
  }, [isReady, regionsPlugin, wavesurfer, highlightedSections]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Prevent default space bar scrolling
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        handlePlayPause();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handlePlayPause]);

  return (
    <div className="flex flex-row items-center w-full h-full">
      <div className="flex justify-center items-center">
        <IconButton
          icon={isPlaying ? FaPause : FaPlay}
          onClick={handlePlayPause}
          colorClass="text-darkpink"
          bgClass="bg-transparent"
          sizeClass="w-10 h-10"
          ariaLabel="play audio"
        />
      </div>
      <div className="flex-grow flex flex-col">
        <div ref={containerRef} className="w-full" />
      </div>
    </div>
  );
};

export default ResponsiveWaveformPlayer;
