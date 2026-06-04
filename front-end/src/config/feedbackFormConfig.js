export const feedbackForm1Config = [
  {
    type: "singleselect",
    question: "Please describe your musical experience level",
    options: ["Beginner", "Intermediate", "Advanced", "Professional"],
    required: true,
  },
  {
    type: "multiselect",
    question: "What genres/styles do you typically play?",
    options: ["Pop", "Rock", "Classical", "Jazz", "Folk", "Country", "Latin"],
    required: true,
  },
  {
    type: "multiselect",
    question: "What instruments do you play?",
    options: [
      "Guitar",
      "Piano",
      "Voice",
      "Violin/viola/cello",
      "Flute/sax/trumpet/clarinet",
      "Bass",
      "Drums/percussion",
    ],
    required: true,
  },
  {
    type: "singleselect",
    question: "How do you currently practice?",
    options: [
      "At a conservatory",
      "At a music school or program",
      "With a private teacher",
      "I'm self-taught",
      "I'm not actively learning right now",
    ],
    required: true,
  },
  {
    type: "singleselect",
    question: "How many hours do you practice per week?",
    options: ["< 1h", "1-2h", "2-5h", "5-10h", "10-15h", "> 15h"],
    required: true,
  },
  {
    type: "singleselect",
    question: "Do you record yourself while practicing?",
    options: ["Usually", "Sometimes", "Never"],
    required: true,
  },
  {
    type: "textAnswer",
    question:
      "When you watch / listen to your recordings to reflect on them, what do you pay attention to?",
    required: true,
  },
  {
    type: "singleselect",
    question:
      "Have you used automatic performance analysis or visualiser tools before?",
    options: ["Yes", "No", "I'm not sure"],
    required: true,
  },
];

export const feedbackForm2Config = [
  {
    type: "statementRating",
    question: "To what extent do you agree with the following statements?",
    statements: [
      "I learned new things about my musical performance when using the MuSA tool.",
      "The MuSA feedback helped me identify where I was/was not achieving desired effects.",
      "I would use a tool like this in my practice.",
      "I understood the provided feedback.",
    ],
    scaleLabels: ["Strongly disagree", "", "", "", "", "Strongly agree"],
    required: true,
  },
  {
    type: "textAnswer",
    question:
      "Which features of the MuSA performance analyzer were most helpful for you? Why?",
    required: false,
  },
  {
    type: "textAnswer",
    question:
      "Were there any features of the MuSA performance analyzer that didn't work for you? If so, what issues did you encounter?",
    required: false,
  },
  {
    type: "textAnswer",
    question:
      "What improvements would enhance your experience? What features would you add?",
    required: false,
  },
  {
    type: "multiscale",
    question:
      "Rate how helpful MuSA was in providing feedback on each of the musical aspects:",
    options: [
      "Pitch",
      "Intonation",
      "Loudness",
      "Dynamics",
      "Timbre",
      "Tempo",
      "Rhythm",
      "Articulation",
      "Style",
    ],
    scaleLabels: ["Not helpful", "", "", "", "", "Very helpful"],
    required: false,
  },
];

export const feedbackForm3Config = [
  {
    type: "singleselect",
    question:
      "Are you aware of any other technologies that help you reflect on your music practice? (e.g., apps or online resources)",
    options: ["Yes", "No"],
    required: false,
  },
  {
    type: "textAnswer",
    question:
      "What other technologies are you aware of that help with reflecting on your music practice?",
    required: false,
  },
  {
    type: "textAnswer",
    question: "What do you like about these solutions?",
    required: false,
  },
  {
    type: "textAnswer",
    question: "What do you dislike about them?",
    required: false,
  },
  {
    type: "textAnswer",
    question:
      "Could you share the approximate pricing structure or cost range for the technologies you mentioned? (e.g., $15/month subscription, one-time purchase, usage-based pricing, etc.)",
    required: false,
  },
  {
    type: "textAnswer",
    question:
      "If you could imagine any future technology or tool to deepen your understanding of your music practice and enhance self-reflection, what might it look like? (e.g., AI-driven performance analytics, wearable devices for real-time feedback, interactive practice journals, etc.)",
    required: false,
  },
  {
    type: "textAnswer",
    question:
      "What price range would you consider reasonable for a technology that helps you reflect on your music practice and performance? (e.g., 20 EUR/month, a one-time 300 USD purchase, etc.)",
    required: false,
  },
];

export const feedbackFormDemoConfig = [
  {
    type: "singleselect",
    question: "Please describe your musical experience level",
    options: [
      "No background",
      "Beginner",
      "Intermediate",
      "Advanced",
      "Professional",
    ],
    required: true,
  },
  {
    type: "multiselect",
    question: "What genres/styles do you typically play?",
    options: [
      "Pop",
      "Rock",
      "Classical",
      "Jazz",
      "Folk",
      "Country",
      "I don't play",
    ],
    required: true,
  },
  {
    type: "multiselect",
    question: "What instruments do you play?",
    options: [
      "Guitar",
      "Piano",
      "Voice",
      "Violin/viola/cello",
      "Flute/sax/trumpet/clarinet",
      "Bass",
      "Drums/percussion",
      "I don't play",
    ],
    required: true,
  },
  {
    type: "multiselect",
    question: "How do you typically practice?",
    options: [
      "In person vocal coaching lessons",
      "Group singing e.g. choir, quartet",
      "Independent practice e.g. home",
      "With a band(s)",
      "Recording videos/audio of myself",
      "I don't play",
    ],
    required: true,
  },
  {
    type: "singleselect",
    question:
      "Have you used automatic vocal analysis or visualiser tools before?",
    options: ["Yes", "No", "I'm not sure"],
    required: true,
  },

  {
    type: "statementRating",
    question: "To what extent do you agree with the following statements?",
    statements: [
      "I learned new things about the voice when exploring this tool.",
      "The visual feedback helped me identify what was happening in the recordings.",
      "I would use a tool like to learn, support my practice, or recommend would it to others who practice.",
      "I thought the system was easy to use.",
    ],
    scaleLabels: ["Strongly disagree", "", "", "", "", "Strongly agree"],
    required: true,
  },
  {
    type: "textAnswer",
    question:
      "If you could imagine any future technology or tool to enhance music practice, what might it look like? (e.g., AI-driven performance analytics, wearable devices for real-time feedback, interactive practice journals, etc.)?",
    required: false,
  },
];
