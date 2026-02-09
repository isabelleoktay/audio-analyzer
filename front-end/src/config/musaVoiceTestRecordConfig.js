export const musaVoiceTestRecordConfig = [
  {
    task: "Pitch Modulation Control",
    title: "Pitch Modulation Exercise",
    instructions: {
      pre: "Your task is to match the reference audio's placement of straight and vibrato vocal techniques. You can sing the same phrase in any key that feels natural to you.",
      post: "To end this section, we ask you to record a final attempt to match the reference audio.",
    },
    techniques: ["straight", "vibrato"],
    highlightLabel: "vibrato",
    defaultLabel: "straight",
    highlightLabelColor: "text-darkpink",
    defaultLabelColor: "text-warmyellow",
    conditions: {
      control: {
        phrase: "Let it be, let it be, let it be, let it be.",
        audio: "/audio/reference/pitch_mod_control.wav",
        highlightedText: [
          { phrase: "let it be,", occurrences: [2] },
          { phrase: "let it be." },
        ],
        highlightClass: "bg-darkpink text-blueblack",
        defaultClass: "bg-warmyellow text-blueblack",
      },
      tool: {
        phrase: "And I-I will always love you.",
        audio: "/audio/reference/pitch_mod_tool.wav",
        highlightedText: [{ phrase: "-I" }, { phrase: "ways love" }],
        highlightClass: "bg-darkpink text-blueblack",
        defaultClass: "bg-warmyellow text-blueblack",
      },
    },
  },

  {
    task: "Vocal Tone Control",
    title: "Vocal Tone Exercise",
    instructions: {
      pre: "Your task is to match the reference audio's placement of belt and breathy vocal techniques. You can sing the same phrase in any key that feels natural to you.",
      post: "To end this section, we ask you to record a final attempt to match the reference audio.",
    },
    techniques: ["belt", "breathy"],
    highlightLabel: "breathy",
    defaultLabel: "belt",
    highlightLabelColor: "text-purple-400",
    defaultLabelColor: "text-electricblue",
    conditions: {
      tool: {
        phrase:
          "Hello... it's me. I was wondering if after all these years you'd like to meet.",
        audio: "/audio/reference/vocal_tone_tool.wav",
        highlightedText: [
          { phrase: "Hello..." },
          { phrase: "I was" },
          { phrase: "if after all these" },
          { phrase: "meet." },
        ],
        highlightClass: "bg-purple-400 text-blueblack",
        defaultClass: "bg-electricblue text-blueblack",
      },
      control: {
        phrase:
          "We are the champions, my friend. And we'll keep on fighting till the end.",
        audio: "/audio/reference/vocal_tone_control.wav",
        highlightedText: [
          { phrase: "the champions," },
          { phrase: "and we'll" },
          { phrase: "till the end." },
        ],
        highlightClass: "bg-purple-400 text-blueblack",
        defaultClass: "bg-electricblue text-blueblack",
      },
    },
  },
];
