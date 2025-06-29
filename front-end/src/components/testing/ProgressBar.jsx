const ProgressBar = ({ currentStep, totalSteps }) => {
  const progressPercentage = Math.round((currentStep / totalSteps) * 100);

  return (
    <div className="fixed bottom-0 w-full">
      {/* Label above the progress bar */}
      <div className="text-center font-bold text-lightpink mb-2">
        Progress Bar
      </div>
      {/* Progress bar */}
      <div className="w-full bg-lightgray">
        <div
          className="h-4 bg-darkpink transition-all"
          style={{ width: `${progressPercentage}%` }}
        ></div>
      </div>
    </div>
  );
};

export default ProgressBar;
