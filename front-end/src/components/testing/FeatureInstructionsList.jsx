const FeatureInstructionsList = () => {
  return (
    <ol className="flex flex-col space-y-4 text-lightpink ml-20">
      <li className="flex flex-col items-start justify-start space-y-1">
        <div className="font-bold text-darkpink text-lg">Pitch Accuracy</div>
        <div>
          Sing the phrase as closely as possible to the correct pitch shown in
          the reference audio.
        </div>
      </li>
      <li className="flex flex-col items-start justify-start space-y-1">
        <div className="font-bold text-darkpink text-lg">Constant Dynamics</div>
        <div>
          Sing the phrase with consistent loudness (volume) from start to
          finish.
        </div>
      </li>
      <li className="flex flex-col items-start justify-start space-y-1">
        <div className="font-bold text-darkpink text-lg">Constant Tempo</div>
        <div>
          Sing the phrase at a steady speed, matching the tempo of the reference
          audio.
        </div>
      </li>
    </ol>
  );
};

export default FeatureInstructionsList;
