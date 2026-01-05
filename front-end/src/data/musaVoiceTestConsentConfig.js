export const musaVoiceTestConsentConfig = [
  {
    title: "Welcome to the Audio Analyzer Test",
    textparts: [
      [
        {
          text: "By participating in this study, you acknowledge that audio recordings will be collected and used for educational and scientific research purposes.",
          bold: false,
        },
      ],
      [
        {
          text: "The data gathered will be limited to the audio itself and computed features derived from the recordings. These features may include, but are not limited to, vocal technique, dynamics, pitch, vibrato, and tempo. All data will be stored securely and used solely for research and educational analysis, with no commercial intent.",
          bold: false,
        },
      ],
      [
        {
          text: "No personal or identifying information will be collected.",
          bold: true,
        },
      ],
      [
        {
          text: "Participation is voluntary, and you may withdraw at any time without consequence.",
          bold: false,
        },
      ],
      [{ text: "Do you consent to participate in this study?", bold: false }],
    ],
    buttonLabels: ["Yes, I Consent", "No, Take Me Back"],
  },
  {
    title: "",
    textparts: [
      [
        {
          text: "The study consists of two separate tasks to test the effectiveness of the tool in providing feedback for different kinds of vocal techniques.",
          bold: false,
        },
      ],
      [
        {
          text: "One task is on controlling pitch modulation techniques (e.g. straight VS vibrato technique), while the other is on controlling vocal tone (e.g. belt VS breathy tone). You can choose to participate in both branches of the study or randomly be allocated to one of them.",
          bold: false,
        },
      ],
      [
        {
          text: "Each task requires a maximum of about 25 minutes, meaning that participating in both requires approximately 50 minutes of your time.",
          bold: true,
        },
      ],
      [
        {
          text: "Please choose whether to participate in both or one task according to your availability.",
          bold: false,
        },
      ],
    ],
    buttonLabels: [
      "I want to participate in both",
      "Randomly allocate me to one",
    ],
  },
];
