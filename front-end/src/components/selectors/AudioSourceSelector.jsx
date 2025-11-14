const AudioSourceSelector = ({
  selectedSource,
  onSourceChange,
  hasUploadedFile,
  hasRecordedAudio,
}) => {
  return (
    <div className="flex flex-row w-full h-full rounded-2xl bg-lightgray/10 p-1 relative">
      {/* Sliding Background */}
      <div
        className={`absolute top-1 bottom-1 w-1/2 bg-lightpink/20 rounded-xl transition-all duration-300 ease-out ${
          selectedSource === "upload" ? "left-1" : "right-1"
        }`}
      />

      {/* Upload Audio Option */}
      <div
        className={`flex-1 flex items-center justify-center cursor-pointer transition-all duration-200 relative z-10 ${
          selectedSource === "upload"
            ? "text-lightpink font-medium"
            : "text-lightgray hover:text-lightpink/70"
        }`}
        onClick={() => onSourceChange("upload")}
      >
        <span className="text-sm">select uploaded audio</span>
      </div>

      {/* Record Audio Option */}
      <div
        className={`flex-1 flex items-center justify-center cursor-pointer transition-all duration-200 relative z-10 ${
          selectedSource === "record"
            ? "text-lightpink font-medium"
            : "text-lightgray hover:text-lightpink/70"
        }`}
        onClick={() => onSourceChange("record")}
      >
        <span className="text-sm">select recorded audio</span>
      </div>
    </div>
  );
};

export default AudioSourceSelector;
