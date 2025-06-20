import { useState, useEffect, useCallback, useRef } from "react";

import ConsentCard from "../components/testing/ConsentCard";
import IntroductionSection from "../components/testing/IntroductionSection";
import InstructionsSection from "../components/testing/InstructionsSection";
import RecordAudioSection from "../components/sections/RecordAudioSection";
import AnalysisButtons from "../components/buttons/AnalysisButtons";
import GraphWithWaveform from "../components/visualizations/GraphWithWaveform";
import SecondaryButton from "../components/buttons/SecondaryButton";
import TertiaryButton from "../components/buttons/TertiaryButton";
import TestingCompleted from "../components/testing/TestingCompleted";
import TestingSection from "../components/testing/TestingSection";
import ResponsiveWaveformPlayer from "../components/visualizations/ResponsiveWaveformPlayer";

import { uploadAudioToPythonService, uploadTestSubject } from "../utils/api";

const TEST_FEATURES = ["pitch", "dynamics", "tempo"];

const Testing = ({ setUploadsEnabled }) => {
  const completedGroupsRef = useRef([]);

  const [currentStep, setCurrentStep] = useState("consent");
  const [testGroup, setTestGroup] = useState("feedback");
  const [subjectData, setSubjectData] = useState({});
  const [attemptCount, setAttemptCount] = useState(0);
  const [currentTestFeatureIndex, setCurrentTestFeatureIndex] = useState(0);
  const [feedbackStage, setFeedbackStage] = useState("before");
  const [selectedAnalysisFeature, setSelectedAnalysisFeature] = useState(null);
  const [audioFeatures, setAudioFeatures] = useState({});
  const [analyzeMode, setAnalyzeMode] = useState(false);
  const [remainingTime, setRemainingTime] = useState(600); // 10 minutes in seconds
  const [isProceedButtonEnabled, setIsProceedButtonEnabled] = useState(false);
  const [feedbackToolUsageCount, setFeedbackToolUsageCount] = useState(0);

  const [currentAudioName, setCurrentAudioName] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleConsent = (consentGiven) => {
    if (consentGiven) {
      // Randomize the group
      const randomizedGroup = Math.random() < 0.5 ? "feedback" : "none";
      const newSubjectId = Math.random().toString(36).substring(2, 10);

      const fileName = `subject-${newSubjectId}-${randomizedGroup}-${TEST_FEATURES[currentTestFeatureIndex]}-${attemptCount}.wav`;
      setCurrentAudioName(fileName);
      setTestGroup(randomizedGroup);

      setSubjectData((prevData) => ({
        ...prevData,
        consent: true,
        subjectId: newSubjectId,
        feedbackToolUsageCount: 0,
        [`${testGroup}`]: {},
      }));

      // Move to the instructions step
      setCurrentStep("introduction");
    } else {
      // Redirect to home page
      window.location.href = "/";
    }
  };

  const handleChangeAttemptCount = () => {
    let fileName;
    if (testGroup === "feedback") {
      if (feedbackStage === "during") {
        fileName = `subject-${
          subjectData.subjectId
        }-${testGroup}-${feedbackStage}-${attemptCount + 1}.wav`;
      } else {
        fileName = `subject-${subjectData.subjectId}-${testGroup}-${
          TEST_FEATURES[currentTestFeatureIndex]
        }-${feedbackStage}-${attemptCount + 1}.wav`;
      }
    } else {
      fileName = `subject-${subjectData.subjectId}-${testGroup}-${
        TEST_FEATURES[currentTestFeatureIndex]
      }-${attemptCount + 1}.wav`;
    }

    setCurrentAudioName(fileName);
    setAttemptCount(attemptCount + 1);
  };

  const updateSubjectData = useCallback(async () => {
    let file;
    if (audioBlob) {
      file = new File([audioBlob], currentAudioName, {
        type: "audio/wav",
      });
      setUploadedFile(file);

      const response = await uploadAudioToPythonService(
        file,
        testGroup,
        testGroup === "feedback" ? feedbackStage : null,
        feedbackStage !== "during"
          ? TEST_FEATURES[currentTestFeatureIndex]
          : null
      );
      console.log(response);

      const updatedData = (() => {
        if (testGroup === "none") {
          return {
            ...subjectData,
            [testGroup]: {
              ...(subjectData[testGroup] || {}),
              [TEST_FEATURES[currentTestFeatureIndex]]: {
                ...(subjectData[testGroup]?.[
                  TEST_FEATURES[currentTestFeatureIndex]
                ] || {}),
                [currentAudioName]: {
                  filePath: response.path,
                },
              },
            },
          };
        } else if (testGroup === "feedback") {
          if (feedbackStage === "during") {
            return {
              ...subjectData,
              [testGroup]: {
                ...(subjectData[testGroup] || {}),
                [feedbackStage]: {
                  ...(subjectData[testGroup]?.[feedbackStage] || {}),
                  [currentAudioName]: {
                    filePath: response.path,
                    audioFeatures: audioFeatures,
                  },
                },
              },
            };
          } else {
            return {
              ...subjectData,
              [testGroup]: {
                ...(subjectData[testGroup] || {}),
                [feedbackStage]: {
                  ...(subjectData[testGroup]?.[feedbackStage] || {}),
                  [TEST_FEATURES[currentTestFeatureIndex]]: {
                    ...(subjectData[testGroup]?.[feedbackStage]?.[
                      TEST_FEATURES[currentTestFeatureIndex]
                    ] || {}),
                    [currentAudioName]: {
                      filePath: response.path,
                    },
                  },
                },
              },
            };
          }
        }
      })();

      setSubjectData(updatedData);

      const uploadTestSubjectRes = await uploadTestSubject(
        updatedData.subjectId,
        updatedData
      );
      console.log(uploadTestSubjectRes);
    }
  }, [
    audioBlob,
    currentAudioName,
    testGroup,
    feedbackStage,
    currentTestFeatureIndex,
    subjectData,
    audioFeatures,
  ]);

  const handleAnalyzeNewRecording = () => {
    const newUsageCount =
      currentStep === "feedback"
        ? feedbackToolUsageCount + 1
        : feedbackToolUsageCount;
    if (currentStep === "feedback") {
      setFeedbackToolUsageCount(newUsageCount);
    }
    setAnalyzeMode(false);
    updateSubjectData();
    setAudioBlob(null);
    setAudioUrl(null);
    setUploadedFile(null);
    setSelectedAnalysisFeature(null);
    setAudioFeatures({});
  };

  const handleAnalysisFeatureSelect = (feature) => {
    setSelectedAnalysisFeature(feature);
    setAnalyzeMode(true);
  };

  const handleFinishTestingTool = useCallback(() => {
    const newUsageCount = feedbackToolUsageCount + 1;
    setFeedbackToolUsageCount(newUsageCount);

    // Update subject data with the new usage count
    setSubjectData((prevData) => ({
      ...prevData,
      feedbackToolUsageCount: newUsageCount,
    }));

    updateSubjectData();
    setFeedbackStage("after");
    setCurrentStep("instructions");
    setCurrentTestFeatureIndex(0);
    setAttemptCount(0);
    setAudioBlob(null);
    setAudioUrl(null);

    const fileName = `subject-${subjectData.subjectId}-feedback-after-${
      TEST_FEATURES[currentTestFeatureIndex]
    }-${0}.wav`;
    setCurrentAudioName(fileName);
    setSelectedAnalysisFeature(null);
    setAudioFeatures({});
  }, [
    subjectData.subjectId,
    currentTestFeatureIndex,
    updateSubjectData,
    feedbackToolUsageCount,
  ]);

  const handleSubmitRecording = () => {
    updateSubjectData();
    if (currentStep === "feedback" && feedbackStage === "during") {
      console.log(":D subject data already updated.");
    } else if (currentTestFeatureIndex < TEST_FEATURES.length - 1) {
      // Move to the next feature
      setAttemptCount(0);
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

      // Update completed groups using ref
      completedGroupsRef.current = [...completedGroupsRef.current, testGroup];
      const updatedGroups = completedGroupsRef.current;
      const testAnalyzer = feedbackStage === "before";

      let newAudioName;
      if (
        updatedGroups.includes("none") &&
        updatedGroups.includes("feedback") &&
        feedbackStage === "after"
      ) {
        setCurrentStep("completed");
      } else if (updatedGroups.includes("feedback")) {
        if (testAnalyzer) {
          setFeedbackStage("during");
          setCurrentStep("instructions");
          newAudioName = `subject-${
            subjectData.subjectId
          }-feedback-during-${0}.wav`;
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
      }
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs < 10 ? `0${secs}` : secs}`;
  };

  useEffect(() => {
    let timer;

    if (currentStep === "feedback") {
      // Start the timer when currentStep is "feedback"
      timer = setInterval(() => {
        setRemainingTime((prevTime) => {
          if (prevTime <= 1) {
            clearInterval(timer); // Stop the timer when it reaches 0
            handleFinishTestingTool();
            return 0;
          }
          return prevTime - 1;
        });
      }, 1000);
    }

    // Cleanup the interval when the component unmounts or currentStep changes
    return () => {
      clearInterval(timer);
    };
  }, [currentStep, handleFinishTestingTool]);

  useEffect(() => {
    // Enable the button once when 480 seconds have passed
    if (!isProceedButtonEnabled && remainingTime <= 480) {
      setIsProceedButtonEnabled(true);
    }
  }, [remainingTime, isProceedButtonEnabled]);

  useEffect(() => {
    // disable enabling uploads from main application
    setUploadsEnabled(false);
  }, [setUploadsEnabled]);

  console.log(subjectData);

  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray">
      {currentStep === "consent" && (
        <ConsentCard handleConsent={handleConsent} />
      )}

      {currentStep === "introduction" && (
        <IntroductionSection
          handleNextStep={() => setCurrentStep("instructions")}
        />
      )}

      {currentStep === "instructions" && (
        <InstructionsSection
          testGroup={testGroup}
          feedbackStage={feedbackStage}
          handleNextStep={() => {
            setUploadedFile(null);
            if (testGroup === "feedback" && feedbackStage === "during") {
              setCurrentStep("feedback");
            } else {
              setCurrentStep("testing");
            }
          }}
        />
      )}

      {currentStep === "completed" && (
        <TestingCompleted subjectData={subjectData} />
      )}

      {currentStep === "testing" && (
        <TestingSection
          currentTestFeature={TEST_FEATURES[currentTestFeatureIndex]}
          attemptCount={attemptCount}
          testGroup={testGroup}
        >
          <RecordAudioSection
            testingEnabled={true}
            audioName="Your Recording"
            audioBlob={audioBlob}
            setAudioBlob={setAudioBlob}
            audioUrl={audioUrl}
            setAudioURL={setAudioUrl}
            attemptCount={attemptCount}
            onChangeAttemptCount={handleChangeAttemptCount}
            onSubmitRecording={handleSubmitRecording}
            updateSubjectData={updateSubjectData}
          />
        </TestingSection>
      )}

      {currentStep === "feedback" && (
        <div className="flex flex-col items-center justify-start h-screen text-lightgray w-full space-y-6">
          <div className="flex flex-col items-center justify-self-start space-y-2 mt-20 w-1/2">
            <div className="text-4xl text-electricblue font-bold">
              Visualization Tool
            </div>
            <div>
              You may record yourself and visualize your audio recordings as
              many times as you want within the next 10 minutes, but you must
              use the tool for at least 2 minutes.
            </div>
          </div>

          <div className="flex flex-col items-center justify-center w-1/2 space-y-8">
            <div className="flex flex-col items-start w-full space-y-2">
              <div className="flex flex-row items-end justify-between w-full">
                <div className="text-xl font-semibold">Reference Audio</div>
                <div className="flex flex-row space-x-2 items-end">
                  <div
                    className={`font-semibold text-sm px-4 py-2 bg-lightgray/25 rounded-2xl transition duration-200 ease-in-out ${
                      remainingTime <= 60 ? "text-rose-300" : "text-warmyellow"
                    }`}
                  >
                    Time Remaining: {formatTime(remainingTime)}
                  </div>
                  <TertiaryButton
                    onClick={handleFinishTestingTool}
                    className="whitespace-nowrap text-sm"
                    disabled={!isProceedButtonEnabled}
                  >
                    proceed to next task
                  </TertiaryButton>
                </div>
              </div>
              <div className="h-[100px] w-full bg-lightgray/25 p-4 rounded-3xl">
                <ResponsiveWaveformPlayer
                  audioUrl="/audio/twinkle_twinkle_little_star_g.m4a"
                  highlightedSections={[]}
                />
              </div>
            </div>
            {!analyzeMode && !uploadedFile && (
              <RecordAudioSection
                feedbackStage={feedbackStage}
                testingEnabled={true}
                audioName="Your Recording"
                audioBlob={audioBlob}
                setAudioBlob={setAudioBlob}
                setAudioURL={setAudioUrl}
                attemptCount={attemptCount}
                onChangeAttemptCount={handleChangeAttemptCount}
                onSubmitRecording={handleSubmitRecording}
                updateSubjectData={updateSubjectData}
              />
            )}
          </div>

          {uploadedFile && (
            <div className="flex flex-col items-center justify-center w-full space-y-6">
              <AnalysisButtons
                selectedInstrument="violin"
                uploadedFile={uploadedFile}
                selectedAnalysisFeature={selectedAnalysisFeature}
                onAnalysisFeatureSelect={handleAnalysisFeatureSelect}
                uploadsEnabled={false}
                audioFeatures={audioFeatures}
                setAudioFeatures={setAudioFeatures}
                audioUuid={null}
                setAudioUuid={() => {}}
              />
              <div className="flex flex-col justify-center items-center w-fit space-y-2">
                <div className="text-xl font-semibold text-lightpink mb-1 self-start">
                  Your Recording
                </div>
                <div className="bg-lightgray/25 rounded-3xl w-fit p-8">
                  <GraphWithWaveform
                    key={audioFeatures[selectedAnalysisFeature]?.audioUrl}
                    audioURL={audioFeatures[selectedAnalysisFeature]?.audioUrl}
                    featureData={
                      audioFeatures[selectedAnalysisFeature]?.data || []
                    }
                    selectedAnalysisFeature={selectedAnalysisFeature}
                    audioDuration={
                      audioFeatures[selectedAnalysisFeature]?.duration
                    }
                  />
                </div>
                <div className="flex flex-row self-end space-x-2">
                  <SecondaryButton onClick={handleAnalyzeNewRecording}>
                    analyze new recording
                  </SecondaryButton>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Testing;
