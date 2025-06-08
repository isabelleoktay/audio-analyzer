import { useState } from "react";

import ConsentCard from "../components/cards/ConsentCard";
import IntroductionSection from "../components/testing/IntroductionSection";
import InstructionsSection from "../components/testing/InstructionsSection";
import RecordAudioSection from "../components/sections/RecordAudioSection";
import ResponsiveWaveformPlayer from "../components/visualizations/ResponsiveWaveformPlayer";
import AnalysisButtons from "../components/buttons/AnalysisButtons";

const TEST_FEATURES = ["pitch", "dynamics", "tempo"];

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

  resetAudioData,
}) => {
  const [currentStep, setCurrentStep] = useState("consent");
  const [testGroup, setTestGroup] = useState("none");
  const [subjectData, setSubjectData] = useState({});
  const [attemptCount, setAttemptCount] = useState(0);
  const [completedGroups, setCompletedGroups] = useState([]);
  const [currentTestFeatureIndex, setCurrentTestFeatureIndex] = useState(0);
  const [feedbackStage, setFeedbackStage] = useState("before");

  console.log("attemptCount", attemptCount);

  const [currentAudioName, setCurrentAudioName] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleConsent = (consentGiven) => {
    if (consentGiven) {
      // Randomize the group
      const randomizedGroup = Math.random() < 0.5 ? "feedback" : "none";
      const newSubjectId = Math.random().toString(36).substring(2, 10);

      const fileName = `subject-${newSubjectId}-${randomizedGroup}-${TEST_FEATURES[currentTestFeatureIndex]}-${attemptCount}.wav`;
      setCurrentAudioName(fileName);
      // setTestGroup(randomizedGroup);

      setSubjectData((prevData) => ({
        ...prevData,
        consent: true,
        subjectId: newSubjectId,
        [`${testGroup}`]: {},
      }));

      // Move to the instructions step
      setCurrentStep("introduction");
    } else {
      // Redirect to home page
      window.location.href = "/";
    }
  };

  const handleNextStep = (nextStep) => {
    setCurrentStep(nextStep);
  };

  const handleChangeAttemptCount = () => {
    let fileName;
    if (testGroup === "feedback") {
      fileName = `subject-${subjectData.subjectId}-${testGroup}-${
        TEST_FEATURES[currentTestFeatureIndex]
      }-${feedbackStage}-${attemptCount + 1}.wav`;
    } else {
      fileName = `subject-${subjectData.subjectId}-${testGroup}-${
        TEST_FEATURES[currentTestFeatureIndex]
      }-${attemptCount + 1}.wav`;
    }

    setCurrentAudioName(fileName);
    setAttemptCount(attemptCount + 1);
  };

  const updateSubjectData = () => {
    if (testGroup === "none") {
      setSubjectData((prevData) => ({
        ...prevData,
        [testGroup]: {
          ...(prevData[testGroup] || {}),
          [TEST_FEATURES[currentTestFeatureIndex]]: {
            ...(prevData[testGroup]?.[TEST_FEATURES[currentTestFeatureIndex]] ||
              {}),
            [currentAudioName]: audioBlob,
          },
        },
      }));
    } else if (testGroup === "feedback") {
      setSubjectData((prevData) => ({
        ...prevData,
        [testGroup]: {
          ...(prevData[testGroup] || {}), // Ensure `testGroup` exists
          [feedbackStage]: {
            ...(prevData[testGroup]?.[feedbackStage] || {}), // Ensure `feedbackGroup` exists
            [TEST_FEATURES[currentTestFeatureIndex]]: {
              ...(prevData[testGroup]?.[feedbackStage]?.[
                TEST_FEATURES[currentTestFeatureIndex]
              ] || {}),
              [currentAudioName]: audioBlob, // Add or update the current audio blob
            },
          },
        },
      }));
    }
  };

  const handleSubmitRecording = () => {
    updateSubjectData();
    if (currentTestFeatureIndex < TEST_FEATURES.length - 1) {
      // Move to the next feature
      setAttemptCount(0); // Reset attempts for the next feature
      const newTestFeatureIndex = currentTestFeatureIndex + 1;
      setCurrentTestFeatureIndex(currentTestFeatureIndex + 1);
      setAudioBlob(null);
      setAudioUrl(null);

      const fileName = `subject-${subjectData.subjectId}-${testGroup}-${
        TEST_FEATURES[newTestFeatureIndex]
      }-${0}.wav`;
      setCurrentAudioName(fileName);
    } else {
      console.log("All features completed. Submitting data...");

      setCompletedGroups((prevGroups) => {
        const updatedGroups = [...prevGroups, testGroup];
        const testAnalyzer = feedbackStage === "before" ? true : false;

        let newAudioName;
        if (updatedGroups.includes("feedback")) {
          if (testAnalyzer) {
            setFeedbackStage("after");
            setCurrentStep("feedback");
            newAudioName = `subject-${
              subjectData.subjectId
            }-feedback-after-${0}.wav`;
          } else {
            setTestGroup("none");
            setCurrentStep("instructions");
            newAudioName = `subject-${subjectData.subjectId}-none-${0}.wav`;
          }
          setCurrentTestFeatureIndex(0);
          setAttemptCount(0);
          setAudioBlob(null);
          setAudioUrl(null);
          setCurrentAudioName(newAudioName);
        } else if (updatedGroups.includes("none")) {
          setTestGroup("feedback");
          setCurrentStep("instructions");
          setCurrentTestFeatureIndex(0);
          setAttemptCount(0);
          setAudioBlob(null);
          setAudioUrl(null);
          const newAudioName = `subject-${
            subjectData.subjectId
          }-feedback-${feedbackStage}-${0}.wav`;
          setCurrentAudioName(newAudioName);
        } else {
          setCurrentStep("completed");
        }

        return updatedGroups;
      });
    }
  };

  console.log(subjectData);

  const handleGenerateSubjectId = () => {
    // const newSubjectId = Math.random().toString(36).substring(2, 10);
    // setSubjectId(newSubjectId);
    // const newTestingPart = Math.random() < 0.5 ? "partA" : "partB";
    // setTestingPart(newTestingPart);
    // const newAudioName = `subject-${newSubjectId}-${newTestingPart}-${subjectAnalysisCount}.wav`;
    // setAudioName(newAudioName);
    // setInRecordMode(true);
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
    <div className="flex flex-col items-center justify-center h-screen text-lightgray">
      {currentStep === "consent" && (
        <ConsentCard handleConsent={handleConsent} />
      )}

      {currentStep === "introduction" && (
        <IntroductionSection
          handleNextStep={() => handleNextStep("instructions")}
        />
      )}

      {currentStep === "instructions" && (
        <InstructionsSection
          testGroup={testGroup}
          handleNextStep={() => setCurrentStep("testing")}
        />
      )}

      {currentStep === "testing" && (
        <div className="flex flex-col items-center justify-center h-screen text-lightgray w-1/2 space-y-8">
          <div className="flex flex-col self-start mb-8 space-y-2">
            <div className="text-4xl text-electricblue font-bold">
              {TEST_FEATURES[currentTestFeatureIndex] === "pitch"
                ? "Pitch Accuracy"
                : TEST_FEATURES[currentTestFeatureIndex] === "dynamics"
                ? "Constant Dynamics"
                : "Constant Tempo"}
            </div>
            <div>
              {TEST_FEATURES[currentTestFeatureIndex] === "pitch"
                ? "Sing the phrase as closely as possible to the correct pitch shown in the reference audio."
                : TEST_FEATURES[currentTestFeatureIndex] === "dynamics"
                ? "Sing the phrase with consistent loudness (volume) from start to finish."
                : "Sing the phrase at a steady speed, matching the tempo of the reference audio."}
            </div>
          </div>
          <div className="flex flex-col items-start w-full space-y-2">
            <div className="flex flex-row items-end justify-between w-full">
              <div className="text-xl font-semibold">Reference Audio</div>
              <div className="text-sm text-warmyellow bg-bluegray rounded-2xl px-4 py-2 w-fit">
                <span className="font-bold">attempts remaining:</span>{" "}
                {3 - attemptCount}
              </div>
            </div>
            <div className="h-[100px] w-full bg-lightgray/25 p-4 rounded-3xl">
              <ResponsiveWaveformPlayer highlightedSections={[]} />
            </div>
          </div>
          <RecordAudioSection
            testingEnabled={true}
            audioName={currentAudioName}
            audioBlob={audioBlob}
            setAudioBlob={setAudioBlob}
            setAudioURL={setAudioUrl}
            attemptCount={attemptCount}
            onChangeAttemptCount={handleChangeAttemptCount}
            onSubmitRecording={handleSubmitRecording}
            updateSubjectData={updateSubjectData}
          />
        </div>
      )}

      {currentStep === "feedback" && (
        <div className="flex flex-col items-center justify-center h-screen text-lightgray w-1/2 space-y-8">
          <div className="flex flex-col self-start space-y-2">
            <div className="text-4xl text-electricblue font-bold">
              Visual Feedback Tool
            </div>
            <div>Try out our feedback visualization tool.</div>
          </div>
          <RecordAudioSection
            testingEnabled={true}
            audioName={currentAudioName}
            audioBlob={audioBlob}
            setAudioBlob={setAudioBlob}
            setAudioURL={setAudioUrl}
            attemptCount={attemptCount}
            onChangeAttemptCount={handleChangeAttemptCount}
            onSubmitRecording={handleSubmitRecording}
            updateSubjectData={updateSubjectData}
          />
          <AnalysisButtons selectedInstrument="violin" />
        </div>
      )}
    </div>
    // <div className="flex flex-col items-center justify-start h-screen py-32 text-lightgray">
    //   <div className="flex flex-col gap-2 items-center">
    //     <SecondaryButton
    //       onClick={handleSetTestingEnabled}
    //       isActive={testingEnabled}
    //     >
    //       {`testing ${testingEnabled ? "enabled" : "disabled"}`}
    //     </SecondaryButton>
    //     {testingEnabled && (
    //       <div className="flex flex-col items-center gap-2 justify-center w-full">
    //         <TertiaryButton onClick={handleGenerateSubjectId}>
    //           generate subject id
    //         </TertiaryButton>

    //         {subjectId && (
    //           <div className="mt-20 flex flex-col items-center gap-2">
    //             <div className="mb-4">
    //               <LeftButton
    //                 onClick={() => handleSetTestingPart("partA")}
    //                 active={testingPart === "partA"}
    //                 asButton={true}
    //                 label="part a"
    //               />

    //               <RightButton
    //                 onClick={() => handleSetTestingPart("partB")}
    //                 active={testingPart === "partB"}
    //                 asButton={true}
    //                 label="part b"
    //               />
    //             </div>
    //             <div className="text-xl font-medium">
    //               <span className="font-bold text-warmyellow">subject: </span>
    //               {subjectId}
    //             </div>
    //             <div className="text-xl font-medium">
    //               <span className="font-bold text-electricblue">
    //                 current audio name:{" "}
    //               </span>
    //               {audioName}
    //             </div>
    //           </div>
    //         )}

    //         {subjectAnalyses && (
    //           <>
    //             {Object.keys(subjectAnalyses).map((part) => {
    //               const partData = subjectAnalyses[part];
    //               if (!partData || Object.keys(partData).length === 0)
    //                 return null;
    //               return (
    //                 <div key={part} className="mt-8 w-full">
    //                   <h2 className="text-lg font-semibold mb-2">
    //                     Subject Analyses - {part}
    //                   </h2>
    //                   <table className="min-w-full border-collapse">
    //                     <thead>
    //                       <tr>
    //                         <th className="p-2 border">Audio Name</th>
    //                         <th className="p-2 border">Instrument</th>
    //                         <th className="p-2 border">Features</th>
    //                       </tr>
    //                     </thead>
    //                     <tbody>
    //                       {Object.entries(partData).map(
    //                         ([audioName, analysis]) => (
    //                           <tr key={audioName}>
    //                             <td className="p-2 border">{audioName}</td>
    //                             <td className="p-2 border">
    //                               {analysis.instrument}
    //                             </td>
    //                             <td className="p-2 border">
    //                               {typeof analysis.audioFeatures === "object"
    //                                 ? Object.keys(analysis.audioFeatures).join(
    //                                     ", "
    //                                   )
    //                                 : analysis.audioFeatures}
    //                             </td>
    //                           </tr>
    //                         )
    //                       )}
    //                     </tbody>
    //                   </table>
    //                 </div>
    //               );
    //             })}
    //           </>
    //         )}
    //       </div>
    //     )}
    //   </div>
    // </div>
  );
};

export default Testing;
