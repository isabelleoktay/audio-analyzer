import { useState, useEffect, useCallback, useRef } from "react";

import ConsentCard from "../components/testing/ConsentCard";
import IntroductionSection from "../components/testing/IntroductionSection";
import InstructionsSection from "../components/testing/InstructionsSection";
import RecordAudioSection from "../components/sections/RecordAudioSection";
import GraphWithWaveform from "../components/visualizations/GraphWithWaveform";
import SecondaryButton from "../components/buttons/SecondaryButton";
import TertiaryButton from "../components/buttons/TertiaryButton";
import TestingCompleted from "../components/testing/TestingCompleted";
import TestingSection from "../components/testing/TestingSection";
import ResponsiveWaveformPlayer from "../components/visualizations/ResponsiveWaveformPlayer";
import Rating from "../components/testing/Rating";
import Questionnaire from "../components/testing/Questionnaire";
import Timer from "../components/Timer";
import ProgressBar from "../components/testing/ProgressBar";

import {
  uploadAudioToPythonService,
  uploadTestSubject,
  processFeatures,
} from "../utils/api";

const TEST_FEATURES = ["pitch", "dynamics", "tempo"];
const STEPS = [
  "consent",
  "introduction",
  "instructions",
  "testing",
  "feedback",
  "rating",
  "questionnaire",
  "completed",
];
const TOTAL_STEPS = 27;

const Testing = ({ setUploadsEnabled }) => {
  const completedGroupsRef = useRef([]);

  const [testGroup, setTestGroup] = useState("feedback");
  const [subjectData, setSubjectData] = useState({});
  const [attemptCount, setAttemptCount] = useState(0);

  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [currentTestFeatureIndex, setCurrentTestFeatureIndex] = useState(0);

  const [feedbackStage, setFeedbackStage] = useState("before");
  const [audioFeatures, setAudioFeatures] = useState({});
  const [analyzeMode, setAnalyzeMode] = useState(false);
  const [isProceedButtonEnabled, setIsProceedButtonEnabled] = useState(false);
  const [feedbackToolUsageCount, setFeedbackToolUsageCount] = useState(0);
  const [progressBarIndex, setProgressBarIndex] = useState(0);

  const [currentAudioName, setCurrentAudioName] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const getAudioUrl = useCallback(() => {
    if (testGroup === "feedback") {
      if (TEST_FEATURES[currentTestFeatureIndex] === "pitch") {
        return "/audio/testing/twinkle.wav";
      } else if (TEST_FEATURES[currentTestFeatureIndex] === "dynamics") {
        return "/audio/testing/twinkle_dynamics.wav";
      } else if (TEST_FEATURES[currentTestFeatureIndex] === "tempo") {
        return "/audio/testing/twinkle_tempo.wav";
      } else {
        return "/audio/testing/twinkle.wav";
      }
    } else if (testGroup === "none") {
      if (TEST_FEATURES[currentTestFeatureIndex] === "pitch") {
        return "/audio/testing/mary.wav";
      } else if (TEST_FEATURES[currentTestFeatureIndex] === "dynamics") {
        return "/audio/testing/mary_dynamics.wav";
      } else if (TEST_FEATURES[currentTestFeatureIndex] === "tempo") {
        return "/audio/testing/mary_tempo.wav";
      } else {
        return "/audio/testing/twinkle.wav";
      }
    }
    // Default fallback
    return `/audio/testing/twinkle.wav`;
  }, [testGroup, currentTestFeatureIndex]);

  const testingAudioUrl = getAudioUrl();

  const handleProcessFeature = async (file, feature) => {
    setAudioFeatures({}); // Reset audio features before processing
    const featureResult = await processFeatures(file, feature);
    const featureData = {
      data: featureResult.data,
      sampleRate: featureResult.sample_rate,
      audioUrl: featureResult.audio_url || "",
      duration: featureResult.duration || 0,
    };

    setAudioFeatures(featureData);
  };

  const moveToNextStep = () => {
    setProgressBarIndex((prevIndex) => prevIndex + 1);
    if (testGroup === "none") {
      // Skip feedback step entirely for "none" group
      if (STEPS[currentStepIndex] === "questionnaire") {
        // Move to "completed" step
        setCurrentStepIndex(7);
      } else if (STEPS[currentStepIndex] === "introduction") {
        setCurrentStepIndex(2);
      } else if (STEPS[currentStepIndex] === "instructions") {
        // Move directly to testing for the first feature
        setCurrentStepIndex(3); // Move to "testing"
      } else if (STEPS[currentStepIndex] === "testing") {
        // Move directly to testing for the first feature
        setCurrentStepIndex(5); // Move to "testing"
      } else if (STEPS[currentStepIndex] === "rating") {
        // Move to the next feature
        if (currentTestFeatureIndex < TEST_FEATURES.length - 1) {
          setAttemptCount(0);
          setCurrentTestFeatureIndex((prevIndex) => prevIndex + 1);
          setCurrentStepIndex(3);
        } else {
          completedGroupsRef.current.push("none");

          if (!completedGroupsRef.current.includes("feedback")) {
            // Switch to "feedback" group and start instructions
            setTestGroup("feedback");
            setCurrentStepIndex(2); // Move to "instructions"
            setCurrentTestFeatureIndex(0); // Reset feature index
            setAttemptCount(0); // Reset attempt count
          } else {
            // Both groups completed, move to "questionnaire"
            setCurrentStepIndex(6);
          }
        }
      }
    } else if (testGroup === "feedback") {
      if (STEPS[currentStepIndex] === "questionnaire") {
        // Move to "completed" step
        setCurrentStepIndex(7);
      } else if (STEPS[currentStepIndex] === "introduction") {
        setCurrentStepIndex(2); // move to instructions
      } else if (STEPS[currentStepIndex] === "instructions") {
        // Move to testing (before) after instructions
        setCurrentStepIndex(3); // Move to "testing"
        setFeedbackStage("before");
      } else if (
        (feedbackStage === "before" || feedbackStage === "after") &&
        STEPS[currentStepIndex] === "testing"
      ) {
        // Move to "during" (feedback tool usage)
        setCurrentStepIndex(5); // Move to "rating"
      } else if (
        STEPS[currentStepIndex] === "rating" &&
        feedbackStage === "before"
      ) {
        setFeedbackStage("during");
        setCurrentStepIndex(4); // Move to "feedback"
      } else if (feedbackStage === "during") {
        // Move to "after" (testing again for the same feature)
        setFeedbackStage("after");
        setCurrentStepIndex(3); // Move back to "testing"
      } else if (
        STEPS[currentStepIndex] === "rating" &&
        feedbackStage === "after"
      ) {
        // Move to the next feature
        if (currentTestFeatureIndex < TEST_FEATURES.length - 1) {
          setCurrentTestFeatureIndex((prevIndex) => prevIndex + 1);
          setFeedbackStage("before");
          setCurrentStepIndex(3); // stay on testing
        } else {
          // Mark "feedback" group as completed
          completedGroupsRef.current.push("feedback");

          if (!completedGroupsRef.current.includes("none")) {
            // Switch to "none" group and start instructions
            setTestGroup("none");
            setCurrentStepIndex(2); // Move to "instructions"
            setCurrentTestFeatureIndex(0); // Reset feature index
            setAttemptCount(0); // Reset attempt count
          } else {
            // Both groups completed, move to "completed"
            setCurrentStepIndex(6);
          }
        }
      }
    }
  };

  const handleConsent = (consentGiven) => {
    if (consentGiven) {
      setProgressBarIndex((prev) => prev + 1);
      // Randomize the group
      const randomizedGroup = Math.random() < 0.5 ? "feedback" : "none";
      const newSubjectId = Math.random().toString(36).substring(2, 10);

      const fileName = `subject-${newSubjectId}-${randomizedGroup}-${
        randomizedGroup === "feedback" ? `${feedbackStage}-` : ""
      }${TEST_FEATURES[currentTestFeatureIndex]}-${attemptCount}.wav`;
      setCurrentAudioName(fileName);
      setTestGroup(randomizedGroup);

      setSubjectData((prevData) => ({
        ...prevData,
        consent: true,
        subjectId: newSubjectId,
        [`${testGroup}`]: {},
      }));

      // Move to the instructions step
      setCurrentStepIndex(1);
    } else {
      // Redirect to home page
      window.location.href = "/";
    }
  };

  const handleChangeAttemptCount = () => {
    const fileName = `subject-${subjectData.subjectId}-${testGroup}-${
      testGroup === "feedback" ? `${feedbackStage}-` : ""
    }${TEST_FEATURES[currentTestFeatureIndex]}-${attemptCount + 1}.wav`;

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

      if (feedbackStage === "during") {
        handleProcessFeature(file, TEST_FEATURES[currentTestFeatureIndex]);
      }

      const response = await uploadAudioToPythonService(
        file,
        testGroup,
        testGroup === "feedback" ? feedbackStage : null,
        TEST_FEATURES[currentTestFeatureIndex]
      );

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
                  ...(feedbackStage === "during" && {
                    feedbackToolUsageCount: feedbackToolUsageCount, // Only add if feedbackStage is "during"
                  }),
                  [currentAudioName]: {
                    filePath: response.path,
                  },
                },
              },
            },
          };
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
    feedbackToolUsageCount,
  ]);

  const handleAnalyzeNewRecording = () => {
    setAudioFeatures({});

    if (STEPS[currentStepIndex] === "feedback") {
      setIsProceedButtonEnabled(true);
    }
    setAnalyzeMode(false);
    updateSubjectData();
    setAudioBlob(null);
    setAudioUrl(null);
    setUploadedFile(null);
  };

  const handleFinishTestingTool = useCallback(() => {
    setProgressBarIndex((prevIndex) => prevIndex + 1);
    updateSubjectData();
    setFeedbackStage("after");
    setCurrentStepIndex(3);
    setAttemptCount(0);
    resetRecordingState();
    setAnalyzeMode(false);

    const fileName = `subject-${subjectData.subjectId}-feedback-after-${
      TEST_FEATURES[currentTestFeatureIndex]
    }-${0}.wav`;
    setCurrentAudioName(fileName);
    setAudioFeatures({});
  }, [subjectData.subjectId, currentTestFeatureIndex, updateSubjectData]);

  const resetRecordingState = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setUploadedFile(null);
    setAudioFeatures({});
    setIsProceedButtonEnabled(false);
    setFeedbackToolUsageCount(0);
  };

  const updateAudioFileName = () => {
    const fileName = `subject-${subjectData.subjectId}-${testGroup}-${
      testGroup === "feedback" ? `${feedbackStage}-` : ""
    }${TEST_FEATURES[currentTestFeatureIndex]}-${attemptCount}.wav`;
    setCurrentAudioName(fileName);
  };

  const handleSubmitRecording = () => {
    if (feedbackStage === "during") {
      setFeedbackToolUsageCount((prev) => prev + 1);
    }
    // Update subject data
    console.log("updating subject data...");
    updateSubjectData();

    // Move to the next step d
    if (feedbackStage !== "during") {
      moveToNextStep();
      // Reset recording-related states
      resetRecordingState();
    }

    // Update the audio file name for the next step
    updateAudioFileName();
  };

  const handleQuestionnaireSubmit = (answers) => {
    // Update subject data with the questionnaire answers
    setSubjectData((prevData) => ({
      ...prevData,
      questionnaireAnswers: answers, // Add answers at the highest level
    }));

    // Move to the next step
    moveToNextStep();
  };

  const handleRatingSubmit = (response) => {
    // Update subject data with the performance rating response
    setSubjectData((prevData) => {
      const updatedData = {
        ...prevData,
        [testGroup]: {
          ...(prevData[testGroup] || {}),
          ...(testGroup === "feedback"
            ? {
                [feedbackStage]: {
                  ...(prevData[testGroup]?.[feedbackStage] || {}),
                  [TEST_FEATURES[currentTestFeatureIndex]]: {
                    ...(prevData[testGroup]?.[feedbackStage]?.[
                      TEST_FEATURES[currentTestFeatureIndex]
                    ] || {}),
                    performanceRating: response, // Add the response for feedback group
                  },
                },
              }
            : {
                [TEST_FEATURES[currentTestFeatureIndex]]: {
                  ...(prevData[testGroup]?.[
                    TEST_FEATURES[currentTestFeatureIndex]
                  ] || {}),
                  performanceRating: response, // Add the response for none group
                },
              }),
        },
      };
      return updatedData;
    });

    // Move to the next step
    setAttemptCount(0);
    moveToNextStep();
  };

  const handleTimerFinish = () => {
    handleFinishTestingTool();
  };

  useEffect(() => {
    // disable enabling uploads from main application
    setUploadsEnabled(false);
  }, [setUploadsEnabled]);

  useEffect(() => {
    console.log("Audio features updated:", audioFeatures);
  }, [audioFeatures]);

  console.log(subjectData);

  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray">
      {STEPS[currentStepIndex] === "consent" && (
        <ConsentCard handleConsent={handleConsent} />
      )}

      {STEPS[currentStepIndex] === "introduction" && (
        <IntroductionSection
          handleNextStep={moveToNextStep}
          subjectData={subjectData}
        />
      )}

      {STEPS[currentStepIndex] === "instructions" && (
        <InstructionsSection
          testGroup={testGroup}
          feedbackStage={feedbackStage}
          handleNextStep={moveToNextStep}
        />
      )}

      {STEPS[currentStepIndex] === "rating" && (
        <Rating
          onSubmit={handleRatingSubmit}
          currentTestFeature={TEST_FEATURES[currentTestFeatureIndex]}
          testGroup={testGroup}
          feedbackStage={feedbackStage}
          completedGroupsRef={completedGroupsRef}
        />
      )}

      {STEPS[currentStepIndex] === "questionnaire" && (
        <Questionnaire onSubmit={handleQuestionnaireSubmit} />
      )}

      {STEPS[currentStepIndex] === "completed" && (
        <TestingCompleted subjectData={subjectData} />
      )}

      {STEPS[currentStepIndex] === "testing" && (
        <TestingSection
          currentTestFeature={TEST_FEATURES[currentTestFeatureIndex]}
          attemptCount={attemptCount}
          testGroup={testGroup}
          feedbackStage={feedbackStage}
          completedGroupsRef={completedGroupsRef}
          audioUrl={testingAudioUrl}
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

      {STEPS[currentStepIndex] === "feedback" && (
        <div className="flex flex-col items-center justify-start h-screen text-lightgray w-full space-y-6">
          <div className="flex flex-col items-center justify-self-start space-y-2 mt-20 w-1/2">
            <div className="text-4xl text-electricblue font-bold capitalize">
              Visualization Tool - {TEST_FEATURES[currentTestFeatureIndex]}
            </div>
            <div className="text-justify">
              Record yourself and visualize your recordings following the
              reference audio within the next 5 minutes. You make sing each note
              in the reference audio on a consonant sound (la, na, etc.). You do
              not have to use up all of the five minutes, but you must analyze
              at least one recording.{" "}
              <span className="font-bold text-lightpink">
                You may use the tool to gain insights on your{" "}
                {TEST_FEATURES[currentTestFeatureIndex]}
              </span>
              .
            </div>
          </div>

          <div className="flex flex-col items-center justify-center w-1/2 space-y-8">
            <div className="flex flex-col items-start w-full space-y-2">
              <div className="flex flex-row items-end justify-between w-full">
                <div className="text-xl font-semibold">Reference Audio</div>
                <div className="flex flex-row space-x-2 items-end">
                  <Timer onTimerFinish={handleTimerFinish} />
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
                  audioUrl={testingAudioUrl}
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
              <div className="flex flex-col justify-center items-center w-fit space-y-2">
                <div className="text-xl font-semibold text-lightpink mb-1 self-start">
                  Your Recording
                </div>
                <div className="bg-lightgray/25 rounded-3xl w-fit p-8">
                  <GraphWithWaveform
                    key={audioFeatures?.audioUrl}
                    audioURL={audioFeatures?.audioUrl}
                    featureData={audioFeatures?.data || []}
                    selectedAnalysisFeature={
                      TEST_FEATURES[currentTestFeatureIndex]
                    }
                    audioDuration={audioFeatures?.duration}
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
      <ProgressBar
        currentStep={progressBarIndex + 1}
        totalSteps={TOTAL_STEPS}
      />
    </div>
  );
};

export default Testing;
