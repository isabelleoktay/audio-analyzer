const musaVoiceSurveyConfig = [
  {
    type: "singleselect",
    question: "How much singing experience do you have?",
    options: ["Beginner", "Intermediate", "Advanced", "Professional"],
    required: true,
  },
  {
    type: "multiselect",
    question: "How do you typically practice?",
    options: [
      "In person vocal coaching lessons",
      "Group singing e.g. choir, quartet",
      "Independent practice e.g. at home",
      "With a band(s)",
      "Recording videos/audio of myself",
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
    question:
      "Have you used automatic vocal analysis or visualiser tools before?",
    options: ["Yes", "No", "I'm not sure"],
    required: true,
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
    required: true,
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
    required: true,
  },
  {
    type: "statementRating",
    question:
      "Rate how useful it would be for you to receive feedback on whether your singing has the following qualities:",
    statements: [
      "How steady or expressive your pitch sounds (e.g. how much it wavers or uses vibrato)",
      "The overall character of your voice (e.g. powerful, soft, breathy)",
      "Use of stylistic vocal sounds (e.g. rough, buzzing, or warm effects)",
      "Which part of your voice resonance you are using (e.g. lower vs higher voice)",
      "Overall sound quality (e.g. nasal, clear, airy)",
    ],
    scaleLabels: ["Not useful", "", "", "", "", "Very useful"],
    required: true,
  },
  {
    type: "singleselect",
    question: "Will you be in a quiet room while using the tool?",
    options: ["Yes", "No"],
    required: true,
  },
  {
    type: "singleselect",
    question: "Are you using headphones while using the tool?",
    options: ["Yes", "No"],
    required: true,
  },
];

export default musaVoiceSurveyConfig;
