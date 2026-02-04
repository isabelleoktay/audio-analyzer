export const EntryQuestionsConfig = [
  {
    type: "singleselect",
    question: "How much singing experience do you have?",
    options: ["Beginner", "Intermediate", "Advanced", "Professional"],
  },
  {
    type: "multiselect",
    question: "How do you typically practice?",
    options: [
      "In person vocal coaching lessons",
      "Group singing e.g. choir, quartet",
      "Using an application / tool",
      "Independent practice e.g. home",
      "With a band(s)",
      "Recording videos/audio of myself",
    ],
  },
  {
    type: "singleselect",
    question: "How many hours do you practice per week?",
    options: ["< 1h", "1-2h", "2-5h", "5-10h", "10-15h", "> 15h"],
  },
  {
    type: "singleselect",
    question:
      "Have you used automatic vocal analysis or visualiser tools before?",
    options: ["Yes", "No", "I'm not sure"],
  },
  {
    type: "multiselect",
    question: "What genres/styles do you typically sing?",
    options: [
      "Pop",
      "Rock",
      "RnB",
      "Soul",
      "Classical/Opera",
      "Jazz",
      "Musical Theatre",
      "Folk",
      "Country",
      "Hip Hop",
      "Heavy metal",
    ],
  },
  {
    type: "singleselect",
    question: "What is your primary voice type?",
    options: [
      "Bass",
      "Baritone",
      "Tenor",
      "Alto",
      "Mezzo",
      "Soprano",
      "I'm not sure",
    ],
  },
  {
    type: "multiscale",
    question:
      "Rate how useful it would be for you to get feedback (presence/absence of them) on the following vocal techniques:",
    options: [
      "Vibrato",
      "Trill",
      "Trillo",
      "Straight tone",
      "Belting",
      "Breathiness",
      "Speech tone",
      "Lip trill",
      "Vocal fry",
      "Inhaled singing",
      "Glissando",
      "Mixed voice",
      "Head voice",
      "Falsetto",
      "Pharyngeal",
    ],
    scaleLabels: ["Not useful", "", "", "", "", "Very useful"],
  },
  {
    type: "singleselect",
    question: "Will you be in a quiet room while using the tool?",
    options: ["Yes", "No"],
  },
  {
    type: "singleselect",
    question: "Are you using headphones while using the tool?",
    options: ["Yes", "No"],
  },
];

export const SurveyBeforePracticeConfig = [
  {
    task: "Pitch Modulation Control",
    usesTool: false,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your pitch modulation (vibrato VS straight tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
    ],
  },
  {
    task: "Pitch Modulation Control",
    usesTool: true,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your pitch modulation (vibrato VS straight tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
    ],
  },
  {
    task: "Vocal Tone Control",
    usesTool: false,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your vocal tone (belt VS breathy tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
    ],
  },
  {
    task: "Vocal Tone Control",
    usesTool: true,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your vocal tone (belt VS breathy tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
    ],
  },
];

export const SurveyAfterPracticeConfig = [
  {
    task: "Pitch Modulation Control",
    usesTool: false,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your pitch modulation (vibrato VS straight tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
      {
        type: "statementRating",
        question:
          "How helpful was the practice round (without the feedback tool) for you to achieve the pitch modulations you desired in the performance?",
        statements: [""],
        scaleLabels: ["Not helpful at all", "", "", "", "", "Very helpful"],
      },
    ],
  },
  {
    task: "Pitch Modulation Control",
    usesTool: true,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your pitch modulation (vibrato VS straight tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
      {
        type: "statementRating",
        question:
          "How helpful was the practice round (with the feedback tool) for you to achieve the pitch modulations you desired in the performance?",
        statements: [""],
        scaleLabels: ["Not helpful at all", "", "", "", "", "Very helpful"],
      },
    ],
  },
  {
    task: "Vocal Tone Control",
    usesTool: false,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your vocal tone (belt VS breathy tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
      {
        type: "statementRating",
        question:
          "How helpful was the practice round (without the feedback tool) for you to achieve the vocal tone you desired in the performance?",
        statements: [""],
        scaleLabels: ["Not helpful at all", "", "", "", "", "Very helpful"],
      },
    ],
  },
  {
    task: "Vocal Tone Control",
    usesTool: true,
    questions: [
      {
        type: "statementRating",
        question:
          "How well do you think you controlled your vocal tone (belt VS breathy tone)?",
        statements: [""],
        scaleLabels: ["Very Poorly", "", "", "", "", "Very well"],
      },
      {
        type: "statementRating",
        question:
          "How helpful was the practice round (with the feedback tool) for you to achieve the vocal tone you desired in the performance?",
        statements: [""],
        scaleLabels: ["Not helpful at all", "", "", "", "", "Very helpful"],
      },
    ],
  },
];

export const FinalExitConfig = [
  {
    section: "Usefulness",
    infoText:
      "Thank you for participating! Please answer the following questions to help us better understand the usefulness of the MuSA vocal technique feedback tool.",
    generalQuestions: [
      {
        type: "textAnswer",
        question:
          "How was your experience with the tool? Did you learn something about your voice?",
      },
      {
        type: "statementRating",
        question: "To what extent do you agree with the following statements?",
        statements: [
          "I learned new things about my voice when exploring this tool.",
          "The visual feedback helped me identify where I was/was not achieving desired effects.",
          "I would use a tool like this in my practice.",
          "I understood what each classifier label meant.",
        ],
        scaleLabels: ["Strongly disagree", "", "", "", "", "Strongly agree"],
      },
      {
        type: "textAnswer",
        question:
          "Where there any classifications that did not make sense? If so, which ones and why?",
      },
      {
        type: "textAnswer",
        question:
          "Which classifications did you trust the most/least? What impacted this?",
      },
    ],
    specificQuestions: [
      {
        type: "multiscale",
        question:
          "Rate how useful it was to get feedback on each of the vocal techniques:",
        options: ["Vibrato", "Straight tone", "Belting", "Breathiness"],
        scaleLabels: ["Not useful", "", "", "", "", "Very useful"],
      },
    ],
  },
  {
    section: "Usability",
    infoText:
      "Almost done! Please answer the following questions to help us better understand the usability of the MuSA vocal technique feedback tool.",
    generalQuestions: [
      {
        type: "statementRating",
        question: "To what extent do you agree with the following statements?",
        statements: [
          "I think I would like to use this system frequently.",
          "I found the system unnecessarily complex.",
          "I thought the system was easy to use.",
          "I think I would need the support of a technical person to be able to use this system.",
          "I found the various functions in this system were well integrated.",
          "I thought there was too much inconsistency in this system.",
          "I would imagine that most people would learn to use this system very quickly.",
          "I found the system very tiring to use.",
          "I felt very confident using the system.",
          "I needed to learn a lot of things before I could get going with this system.",
        ],
        scaleLabels: ["Strongly disagree", "", "", "", "", "Strongly agree"],
      },
      {
        type: "textAnswer",
        question:
          "What further developments/features would you like this tool to have?",
      },
    ],
  },
];
