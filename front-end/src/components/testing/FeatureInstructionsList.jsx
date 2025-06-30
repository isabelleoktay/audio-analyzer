const FeatureInstructionsList = () => {
  return (
    <ol className="flex flex-col space-y-4 text-lightpink ml-0 md:ml-20">
      <li className="flex flex-col items-start justify-start space-y-1">
        <div className="font-bold text-darkpink text-lg">Pitch Accuracy</div>
        <div>
          Sing the musical phrase as closely as possible to the pitch of the
          reference audio.
        </div>
      </li>
      <li className="flex flex-col items-start justify-start space-y-1">
        <div className="font-bold text-darkpink text-lg">Constant Dynamics</div>
        <div>
          Sing the musical phrase following the loudness levels of the reference
          audio.
        </div>
      </li>
      <li className="flex flex-col items-start justify-start space-y-1">
        <div className="font-bold text-darkpink text-lg">Constant Tempo</div>
        <div>
          Sing the musical phrase following the tempo changes of the reference
          audio.
        </div>
      </li>
    </ol>
  );
};

export default FeatureInstructionsList;
