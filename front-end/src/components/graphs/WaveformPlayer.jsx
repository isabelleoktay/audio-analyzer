import { useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions";

const WaveformPlayer = ({
  audioUrl,
  width,
  height,
  highlightedSections,
  wavesurferRef,
  setIsPlaying,
}) => {
  const containerRef = useRef();

  useEffect(() => {
    if (!containerRef.current) return;

    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
      wavesurferRef.current = null;
    }

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#FFD6E8",
      progressColor: "#FF89BB",
      height,
      width,
      responsive: true,
      normalize: true,
      barWidth: 2,
      plugins: [RegionsPlugin.create()],
    });

    wavesurferRef.current = ws;
    ws.load(audioUrl);

    ws.on("ready", () => {
      const regions = ws.getActivePlugins().regions;
      if (regions) {
        highlightedSections.forEach(({ start, end }) => {
          regions.addRegion({
            start,
            end,
            color: "rgba(255, 203, 107, 0.25)",
            drag: false,
            resize: false,
          });
        });
      }
    });

    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    return () => {
      if (wavesurferRef.current) {
        // destroy() may be async and throw AbortError, so handle it as a promise
        Promise.resolve(wavesurferRef.current.destroy()).catch((e) => {
          if (e.name !== "AbortError") {
            // Only log unexpected errors
            console.error(e);
          }
        });
        wavesurferRef.current = null;
      }
    };
  }, [
    audioUrl,
    height,
    width,
    highlightedSections,
    setIsPlaying,
    wavesurferRef,
  ]);

  return (
    <>
      <div ref={containerRef} className="w-full" />
    </>
  );
};

export default WaveformPlayer;
