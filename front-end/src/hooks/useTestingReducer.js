export const initialState = {
  currentStep: "consent",
  testGroup: "none",
  subjectData: {},
  attemptCount: 0,
  completedGroups: [],
  currentTestFeatureIndex: 0,
  feedbackStage: "before",
  currentAudioName: null,
  audioBlob: null,
  audioUrl: null,
};

export const testingReducer = (state, action) => {
  switch (action.type) {
    case "SET_STEP":
      return { ...state, currentStep: action.payload };
    case "SET_TEST_GROUP":
      return { ...state, testGroup: action.payload };
    case "UPDATE_SUBJECT_DATA":
      return { ...state, subjectData: action.payload };
    case "RESET_AUDIO":
      return { ...state, audioBlob: null, audioUrl: null };
    case "INCREMENT_ATTEMPT":
      return { ...state, attemptCount: state.attemptCount + 1 };
    case "RESET_ATTEMPT":
      return { ...state, attemptCount: 0 };
    case "SET_AUDIO_NAME":
      return { ...state, currentAudioName: action.payload };
    case "NEXT_FEATURE":
      return {
        ...state,
        currentTestFeatureIndex: state.currentTestFeatureIndex + 1,
        attemptCount: 0,
        audioBlob: null,
        audioUrl: null,
      };
    default:
      return state;
  }
};
