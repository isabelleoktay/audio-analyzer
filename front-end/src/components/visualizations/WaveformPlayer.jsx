import { useEffect, useRef, useMemo, useCallback } from "react";
import IconButton from "../buttons/IconButton";
import { FaPlay, FaPause } from "react-icons/fa";
import { useWavesurfer } from "@wavesurfer/react";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions";
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline";
import ZoomPlugin from "wavesurfer.js/dist/plugins/zoom.esm.js";

const width = 800;
const leftMargin = 50;
const rightMargin = 20;
const buttonWidth = 40; // w-10 = 40px
const height = 50;

const WaveformPlayer = ({
  audioUrl,
  highlightedSections,
  waveColor = "#FFD6E8",
  progressColor = "#FF89BB",
  startTime,
  endTime,
  audioDuration,
  feature,
  playIconColorClass = "text-darkpink",
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
  const zoomPlugin = useMemo(
    () =>
      ZoomPlugin.create({
        scale: 0.5,
      }),
    []
  );
  const plugins = useMemo(
    () => [regionsPlugin, timelinePlugin, zoomPlugin],
    [regionsPlugin, timelinePlugin, zoomPlugin]
  );

  const calculateInitialZoom = (audioDuration) => {
    if (!audioDuration) return 50; // default

    // Adjust zoom so the full waveform fits nicely in the container
    const containerWidth = width - leftMargin - rightMargin;
    return Math.max(25, Math.min(100, containerWidth / audioDuration));
  };

  const { wavesurfer, isReady, isPlaying } = useWavesurfer({
    container: containerRef,
    height,
    width: width - leftMargin - rightMargin,
    responsive: true,
    normalize: true,
    barWidth: 2,
    waveColor: waveColor,
    progressColor: progressColor,
    url: audioUrl,
    plugins: plugins,
    hideScrollbar: true,
    autoScroll: false,
    minPxPerSec: calculateInitialZoom(audioDuration),
  });

  const handlePlayPause = useCallback(() => {
    if (wavesurfer) {
      // If not playing and we have a start time, seek to it first
      if (!isPlaying && startTime !== undefined && startTime > 0) {
        wavesurfer.setTime(startTime);
      }
      wavesurfer.playPause();
    }
  }, [wavesurfer, isPlaying, startTime]);

  useEffect(() => {
    if (
      isReady &&
      wavesurfer &&
      startTime !== undefined &&
      endTime !== undefined
    ) {
      const duration = wavesurfer.getDuration();

      // Check if we have a specific time range (zoomed) or full range
      const isZoomed = startTime > 0 || endTime < duration;

      if (isZoomed) {
        const zoomDuration = endTime - startTime;
        const containerWidth = width - leftMargin - rightMargin;

        // Calculate zoom level to fit the time range in the container
        const targetMinPxPerSec = containerWidth / zoomDuration;

        // Apply zoom
        wavesurfer.zoom(targetMinPxPerSec);

        // Scroll to the zoomed section
        wavesurfer.setScrollTime(startTime);
      } else {
        // Reset to initial zoom when showing full range
        const initialZoom = calculateInitialZoom(audioDuration);
        wavesurfer.zoom(initialZoom);
        wavesurfer.setTime(0);
      }
    }
  }, [isReady, wavesurfer, startTime, endTime, audioDuration]);

  useEffect(() => {
    if (
      isReady &&
      wavesurfer &&
      startTime !== undefined &&
      endTime !== undefined
    ) {
      const duration = wavesurfer.getDuration();
      const isZoomed = startTime > 0 || endTime < duration;

      if (isZoomed) {
        const handleTimeUpdate = () => {
          const currentTime = wavesurfer.getCurrentTime();
          // Stop playback if we've reached the end of the zoomed range
          if (currentTime >= endTime) {
            wavesurfer.pause();
            wavesurfer.setTime(startTime);
          }
        };

        wavesurfer.on("timeupdate", handleTimeUpdate);
        return () => wavesurfer.un("timeupdate", handleTimeUpdate);
      }
    }
  }, [isReady, wavesurfer, startTime, endTime]);

  useEffect(() => {
    if (isReady) {
      regionsPlugin.clearRegions();

      highlightedSections.forEach(({ start, end }) => {
        regionsPlugin.addRegion({
          color:
            feature === "pitch"
              ? "rgba(255, 137, 187, 0.25)"
              : "rgba(255, 203, 107, 0.25)", // Dynamic color based on feature
          start: start,
          end: end,
          drag: false,
          resize: false,
        });
      });

      regionsPlugin.on("region-clicked", (region, e) => {
        region.setOptions({
          color:
            feature === "pitch"
              ? "rgba(255, 137, 187, 0.5)"
              : "rgba(255, 203, 107, 0.5)", // Highlighted color
        });
        e.stopPropagation(); // prevent triggering a click on the waveform
        region.play(true);
      });

      regionsPlugin.on("region-out", (region) => {
        region.setOptions({
          color:
            feature === "pitch"
              ? "rgba(255, 137, 187, 0.25)"
              : "rgba(255, 203, 107, 0.25)", // Reset color
        });
      });
    }

    return () => {
      if (wavesurfer) {
        regionsPlugin.unAll();
      }
    };
  }, [isReady, regionsPlugin, wavesurfer, highlightedSections, feature]);

  useEffect(() => {
    const container = containerRef.current;

    const preventScroll = (e) => {
      e.preventDefault();
    };

    if (container) {
      container.addEventListener("wheel", preventScroll, { passive: false });
      container.addEventListener("touchmove", preventScroll, {
        passive: false,
      });
      container.addEventListener("touchmove", preventScroll, {
        passive: false,
      });
    }

    return () => {
      if (container) {
        container.removeEventListener("wheel", preventScroll);
        container.removeEventListener("touchmove", preventScroll);
      }
    };
  }, []);

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
          onClick={handlePlayPause}
          colorClass={playIconColorClass}
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
        <div ref={containerRef} className="w-full" />
        <div ref={timelineRef} className="w-full mt-2" />
      </div>
    </div>
  );
};

export default WaveformPlayer;
