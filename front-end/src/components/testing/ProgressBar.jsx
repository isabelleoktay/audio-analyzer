const ProgressBar = ({ currentStep, totalSteps }) => {
  const progressPercentage = Math.round((currentStep / totalSteps) * 100);

  // Change color when complete
  const barColor =
    progressPercentage >= 100 ? "bg-electricblue/50" : "bg-darkpink/50";

  return (
    <div className="w-full z-20 bg-blueblack">
      {/* Progress bar container */}
      <div
        className="w-full bg-darkgray/60 relative rounded-b-md overflow-hidden"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progressPercentage}
      >
        <div
          className={`h-6 transition-all rounded-b-md overflow-hidden ${barColor}`}
          style={{ width: `${progressPercentage}%` }}
        ></div>

        {/* Centered label inside the bar */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="font-bold text-white text-sm">
            {progressPercentage >= 100 ? "Complete" : "Progress Bar"}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ProgressBar;
