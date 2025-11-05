export const feedbackForm1Config = [
  {
    type: "scale",
    question: "Please describe your musical experience level",
    options: ["Beginner", "Intermediate", "Advanced", "Professional"],
  },
  {
    type: "singleselect",
    question: "Please describe your musical experience level",
    options: ["Beginner", "Intermediate", "Advanced", "Professional"],
  },
  {
    type: "multiselect",
    question: "What genres/styles do you typically play?",
    options: ["Pop", "Rock", "Classical", "Jazz", "Folk", "Country", "Latin"],
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
  },
  {
    type: "singleselect",
    question: "How many hours do you practice per week?",
    options: ["< 1h", "1-2h", "2-5h", "5-10h", "10-15h", "> 15h"],
  },
  {
    type: "singleselect",
    question: "Do you record yourself while practicing?",
    options: ["Usually", "Sometimes", "Never"],
  },
  {
    type: "textAnswer",
    question:
      "When you watch / listen to your recordings to reflect on them, what do you pay attention to?",
  },
  {
    type: "singleselect",
    question:
      "Have you used automatic performance analysis or visualiser tools before?",
    options: ["Yes", "No", "I'm not sure"],
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
    scaleLabels: ["Strongly disagree", "", "", "", "Strongly agree"],
  },
  {
    type: "textAnswer",
    question:
      "Which features of the MuSA performance analyzer were most helpful for you? Why?",
  },
  {
    type: "textAnswer",
    question:
      "Were there any features of the MuSA performance analyzer that didn't work for you? If so, what issues did you encounter?",
  },
  {
    type: "textAnswer",
    question:
      "What improvements would enhance your experience? What features would you add?",
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
    scaleLabels: ["Not helpful", "", "", "", "Very helpful"],
  },
];

export const feedbackForm3Config = [
  {
    type: "singleselect",
    question:
      "Are you aware of any other technologies that help you reflect on your music practice? (e.g., apps or online resources)",
    options: ["Yes", "No"],
  },
  {
    type: "textAnswer",
    question:
      "What other technologies are you aware of that help with reflecting on your music practice?",
  },
  {
    type: "textAnswer",
    question: "What do you like about these solutions?",
  },
  {
    type: "textAnswer",
    question: "What do you dislike about them?",
  },
  {
    type: "textAnswer",
    question:
      "Could you share the approximate pricing structure or cost range for the technologies you mentioned? (e.g., $15/month subscription, one-time purchase, usage-based pricing, etc.)",
  },
  {
    type: "textAnswer",
    question:
      "If you could imagine any future technology or tool to deepen your understanding of your music practice and enhance self-reflection, what might it look like? (e.g., AI-driven performance analytics, wearable devices for real-time feedback, interactive practice journals, etc.)",
  },
  {
    type: "textAnswer",
    question:
      "What price range would you consider reasonable for a technology that helps you reflect on your music practice and performance? (e.g., 20 EUR/month, a one-time 300 USD purchase, etc.)",
  },
];
