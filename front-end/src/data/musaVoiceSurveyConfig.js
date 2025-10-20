const musaVoiceSurveyConfig = [
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
      "Independent practice e.g. at home",
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
      "PortaInhaled singing",
      "Glissando",
      "Mixed voice",
      "Head voice",
      "Falsetto",
      "Pharyngeal",
    ],
    minLabel: "Not useful",
    maxLabel: "Very useful",
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

export default musaVoiceSurveyConfig;
