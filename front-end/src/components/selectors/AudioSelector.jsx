const AudioSelector = ({
  selectedOption,
  onSourceChange,
  hasUploadedFile,
  hasRecordedAudio,
  option1 : propOption1,
  option2 : propOption2, 
  option1Text : propOption1Text, 
  option2Text : propOption2Text, 
}) => {

  // Defaults
  const defaultOption1 = "upload";
  const defaultOption2 = "record";
  const defaultOption1Text = "select uploaded audio";
  const defaultOption2Text = "select recorded audio";

  const option1 =
    propOption1 || defaultOption1;
  const option2 =
    propOption2 || defaultOption2;
  const option1Text =
    propOption1Text || defaultOption1Text;
  const option2Text =
    propOption2Text || defaultOption2Text;


  return (
    <div className="flex flex-row w-full h-full rounded-2xl bg-lightgray/10 p-1 relative">
      {/* Sliding Background */}
      <div
        className={`absolute top-1 bottom-1 w-1/2 bg-lightpink/20 rounded-xl transition-all duration-300 ease-out ${
          selectedOption === option1 ? "left-1" : "right-1"
        }`}
      />

      {/* Option 1 */}
      <div
        className={`flex-1 flex items-center justify-center cursor-pointer transition-all duration-200 relative z-10 ${
          selectedOption === option1
            ? "text-lightpink font-medium"
            : "text-lightgray hover:text-lightpink/70"
        }`}
        onClick={() => onSourceChange(option1)}
      >
        <span className="text-sm">{option1Text}</span>
      </div>

      {/* Option 2 */}
      <div
        className={`flex-1 flex items-center justify-center cursor-pointer transition-all duration-200 relative z-10 ${
          selectedOption === option2
            ? "text-lightpink font-medium"
            : "text-lightgray hover:text-lightpink/70"
        }`}
        onClick={() => onSourceChange(option2)}
      >
        <span className="text-sm">{option2Text}</span>
      </div>
    </div>
  );
};

export default AudioSelector;
