const ProgressBar = ({ currentStep, totalSteps }) => {
  const progressPercentage = Math.round((currentStep / totalSteps) * 100);

  return (
    <div className="fixed top-0 w-full z-20 bg-blueblack">
      {/* Progress bar with centered label inside */}
      <div
        className="w-full bg-darkgray/60 relative rounded-b-md overflow-hidden"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progressPercentage}
      >
        <div
          className="h-6 bg-darkpink/50 transition-all rounded-b-md overflow-hidden"
          style={{ width: `${progressPercentage}%` }}
        ></div>
        {/* Centered label inside the bar */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="font-bold text-white text-sm">Progress Bar</span>
        </div>
      </div>
    </div>
  );
};

export default ProgressBar;
