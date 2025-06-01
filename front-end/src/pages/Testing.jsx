import SecondaryButton from "../components/buttons/SecondaryButton";
import TertiaryButton from "../components/buttons/TertiaryButton";
import RightButton from "../components/buttons/RightButton";
import LeftButton from "../components/buttons/LeftButton";

const Testing = ({
  testingEnabled,
  setTestingEnabled,
  subjectId,
  setSubjectId,
  testingPart,
  setTestingPart,
  audioName,
  setAudioName,
  subjectAnalysisCount,
  setSubjectAnalysisCount,
  setInRecordMode,
  subjectAnalyses,
  resetAudioData,
}) => {
  const handleGenerateSubjectId = () => {
    const newSubjectId = Math.random().toString(36).substring(2, 10);
    setSubjectId(newSubjectId);
    const newTestingPart = Math.random() < 0.5 ? "partA" : "partB";
    setTestingPart(newTestingPart);
    const newAudioName = `subject-${newSubjectId}-${newTestingPart}-${subjectAnalysisCount}.wav`;
    setAudioName(newAudioName);
    setInRecordMode(true);
  };

  const handleSetTestingPart = (part) => {
    setTestingPart(part);
    setSubjectAnalysisCount(1);
    const newAudioName = `subject-${subjectId}-${part}-${1}.wav`;
    setAudioName(newAudioName);
    resetAudioData();
    setInRecordMode(true);
  };

  const handleSetTestingEnabled = () => {
    const newValue = !testingEnabled;
    if (!newValue) {
      setSubjectId(null);
      setTestingPart("partA");
      setSubjectAnalysisCount(1);
      setAudioName("untitled.wav");
    }
    setTestingEnabled(newValue);
    resetAudioData();
    setInRecordMode(true);
  };

  return (
    <div className="flex flex-col items-center justify-start h-screen py-32 text-lightgray">
      <div className="flex flex-col gap-2 items-center">
        <SecondaryButton
          onClick={handleSetTestingEnabled}
          isActive={testingEnabled}
        >
          {`testing ${testingEnabled ? "enabled" : "disabled"}`}
        </SecondaryButton>
        {testingEnabled && (
          <div className="flex flex-col items-center gap-2 justify-center w-full">
            <TertiaryButton onClick={handleGenerateSubjectId}>
              generate subject id
            </TertiaryButton>

            {subjectId && (
              <div className="mt-20 flex flex-col items-center gap-2">
                <div className="mb-4">
                  <LeftButton
                    onClick={() => handleSetTestingPart("partA")}
                    active={testingPart === "partA"}
                    asButton={true}
                    label="part a"
                  />

                  <RightButton
                    onClick={() => handleSetTestingPart("partB")}
                    active={testingPart === "partB"}
                    asButton={true}
                    label="part b"
                  />
                </div>
                <div className="text-xl font-medium">
                  <span className="font-bold text-warmyellow">subject: </span>
                  {subjectId}
                </div>
                <div className="text-xl font-medium">
                  <span className="font-bold text-electricblue">
                    current audio name:{" "}
                  </span>
                  {audioName}
                </div>
              </div>
            )}

            {subjectAnalyses && (
              <>
                {Object.keys(subjectAnalyses).map((part) => {
                  const partData = subjectAnalyses[part];
                  if (!partData || Object.keys(partData).length === 0)
                    return null;
                  return (
                    <div key={part} className="mt-8 w-full">
                      <h2 className="text-lg font-semibold mb-2">
                        Subject Analyses - {part}
                      </h2>
                      <table className="min-w-full border-collapse">
                        <thead>
                          <tr>
                            <th className="p-2 border">Audio Name</th>
                            <th className="p-2 border">Instrument</th>
                            <th className="p-2 border">Features</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(partData).map(
                            ([audioName, analysis]) => (
                              <tr key={audioName}>
                                <td className="p-2 border">{audioName}</td>
                                <td className="p-2 border">
                                  {analysis.instrument}
                                </td>
                                <td className="p-2 border">
                                  {typeof analysis.audioFeatures === "object"
                                    ? Object.keys(analysis.audioFeatures).join(
                                        ", "
                                      )
                                    : analysis.audioFeatures}
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  );
                })}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Testing;
