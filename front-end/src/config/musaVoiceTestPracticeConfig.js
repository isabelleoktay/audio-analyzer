export const musaVoiceTestPracticeConfig = [
  {
    task: "Pitch Modulation Control",
    title: "Pitch Modulation",
    practiceTime: 420,
    instructions: {
      withTool:
        "You now will have a maximum of 7 minutes to practice attaining the same vocal technique placement as the reference audio with feedback from the pitch modulation feedback tool.",
      withoutTool:
        "You now will have a maximum of 7 minutes to practice attaining the same vocal technique placement as the reference audio without feedback from the pitch modulation feedback tool.",
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
    title: "Vocal Tone",
    practiceTime: 420,
    instructions: {
      withTool:
        "You now will have a maximum of 7 minutes to practice attaining the same vocal technique placement as the reference audio with feedback from the vocal tone control feedback tool.",
      withoutTool:
        "You now will have a maximum of 7 minutes to practice attaining the same vocal technique placement as the reference audio without feedback from the vocal tone control feedback tool.",
    },
    techniques: ["belt", "breathy"],
    highlightLabel: "breathy",
    defaultLabel: "belt",
    highlightLabelColor: "text-purple-400",
    defaultLabelColor: "text-electricblue",
    conditions: {
      control: {
        phrase:
          "Hello... it's me. I was wondering if after all these years you'd like to meet.",
        audio: "/audio/reference/vocal_tone_control.wav",
        highlightedText: [
          { phrase: "Hello..." },
          { phrase: "I was" },
          { phrase: "if after all these" },
          { phrase: "meet." },
        ],
        highlightClass: "bg-purple-400 text-blueblack",
        defaultClass: "bg-electricblue text-blueblack",
      },
      tool: {
        phrase:
          "We are the champions, my friend. And we'll keep on fighting till the end.",
        audio: "/audio/reference/vocal_tone_tool.wav",
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
