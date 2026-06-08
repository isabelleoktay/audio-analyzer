import React from "react";
import {
  FaMusic,
  FaUpload,
  FaMicrophone,
  FaChartLine,
  FaPlay,
  FaCheck,
  FaBullseye,
} from "react-icons/fa";

const HowToUse = () => {
  const analyzer_steps = [
    {
      icon: FaMusic,
      title: "1. select an instrument",
      description:
        "choose the instrument you want to analyze from the available options (violin, voice, etc.)",
      tips: [
        "make sure to select the instrument that matches your audio recording",
      ],
    },
    {
      icon: FaUpload,
      title: "2. upload your audio",
      description: "upload an audio file or record directly in the browser",
      tips: [
        "supported formats: wav, mp3, m4a",
        "for best results, use monophonic recordings (single notes/voice)",
        "keep recordings under 2 minutes for faster processing",
      ],
    },
    {
      icon: FaMicrophone,
      title: "3. record audio (optional)",
      description:
        "click the record button to capture audio directly from your microphone",
      tips: ["record in a quiet environment for better analysis"],
    },
    {
      icon: FaChartLine,
      title: "4. choose analysis feature",
      description:
        "select which audio feature you want to analyze (pitch, dynamics, etc.)",
      tips: ["processing may take a few moments for longer files"],
    },
    {
      icon: FaPlay,
      title: "5. explore your results",
      description:
        "view the interactive graph and play back your audio with synchronized highlighting",
      tips: [
        "click and drag on the graph to zoom into specific sections",
        "the waveform player shows highlighted sections from your analysis",
        "click on highlighted regions to play specific parts",
      ],
    },
  ];

  const musa_voice_steps = [
    {
      icon: FaUpload,
      title: "1. add reference audio",
      description: "click to upload from device or record audio",
      tips: [
        "click the record button to capture audio directly from your microphone",
        "record in a quiet environment for better analysis",
      ],
    },
    {
      icon: FaMicrophone,
      title: "2. add input audio",
      description: "this will be compared to the reference audio",
      tips: ["same procedure to upload/record as for the reference audio"],
    },
    {
      icon: FaMusic,
      title: "3. set voice type",
      description: "this will be used by the system to calibrate the analysis",
      tips: [
        "best performance when input and reference have the same voice type",
      ],
    },
    {
      icon: FaCheck,
      title: "4. set target techniques",
      description:
        "mark which vocal techniques you tried to emulate in the recording",
    },
    {
      icon: FaChartLine,
      title: "5. choose analysis feature",
      description:
        "select which audio feature you want to analyze (pitch, dynamics, etc.)",
      tips: [
        "you can compare CLAP and Whisper vocal tone and pitch modulation models",
      ],
    },
    {
      icon: FaPlay,
      title: "5. explore your results",
      description:
        "view the interactive graph and play back your audio",
    },
  ];

  const demo_steps = [
    {
      icon: FaBullseye,
      title: "1. choose a reference audio",
      description: "choose from the given presets",
      tips: [
        "you can listen to the preset audios",
        "certain files are examples of the data used for testing the models with same audio characteristics to training",
        "other files provide 'real' examples to explore the tool with",
        "files with 'target' provide labelling on where certain vocal techniques were emulated",
      ],
    },
    {
      icon: FaMicrophone,
      title: "2. choose or record the input audio",
      description:
        "you can record your own audio, trying to emulate the reference audio",
      tips: [
        "have fun with it; record or choose a preset file for exploration",
      ],
    },
    {
      icon: FaChartLine,
      title: "3. choose analysis feature",
      description:
        "select which audio feature you want to analyze (pitch, dynamics, etc.)",
      tips: [
        "you can compare CLAP and Whisper vocal tone and pitch modulation models",
      ],
    },
    {
      icon: FaPlay,
      title: "4. explore your results",
      description:
        "view the interactive graph and play back your audio",
    },
  ];

  const instructionSets = [
    {
      id: "analyzer",
      title: "performance analyzer",
      videoSrc: "videos/MuSA_performance_analyzer_instructions.mp4",
      steps: analyzer_steps,
    },
    {
      id: "musa_voice",
      title: "musa voice",
      videoSrc: "videos/MuSA_voice_instructions.mp4",
      steps: musa_voice_steps,
    },
    {
      id: "demo",
      title: "demo platform",
      videoSrc: "videos/MuSA_demo_instructions.mp4",
      steps: demo_steps,
    },
  ];

  return (
    <div className="min-h-screen justify-center pt-20 flex flex-col">
      {/* Three Column Layout */}
      <div className="w-full flex gap-6 px-6">
        {instructionSets.map((instructionSet) => (
          <div
            key={instructionSet.id}
            className="flex-1 bg-bluegray/25 rounded-3xl p-6 flex flex-col"
          >
            {/* Title */}
            <h2 className="text-2xl font-bold text-lightpink mb-4 text-center">
              {instructionSet.title}
            </h2>

            {/* Video */}
            {/* <div className="w-full mb-6 rounded-lg overflow-hidden">
              <video className="w-full rounded-lg" controls>
                <source src={instructionSet.videoSrc} type="video/mp4" />
              </video>
            </div> */}

            {/* Steps */}
            <div className="space-y-4 flex-1">
              {instructionSet.steps.map((step, index) => (
                <div key={index} className="bg-lightgray/10 rounded-2xl p-4">
                  <div className="flex items-start space-x-3">
                    {/* Icon */}
                    <div className="flex-shrink-0">
                      <div className="w-10 h-10 bg-gradient-to-r from-darkpink to-electricblue rounded-full flex items-center justify-center">
                        <step.icon className="text-lg text-blueblack" />
                      </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1">
                      <h4 className="text-lg font-semibold text-lightpink mb-2">
                        {step.title}
                      </h4>
                      <p className="text-lightgray mb-2 text-xs">
                        {step.description}
                      </p>

                      {/* Tips */}
                      {step.tips && step.tips.length > 0 && (
                        <div className="bg-blueblack/30 rounded-lg p-3 mt-2">
                          <ul className="space-y-1">
                            {step.tips.map((tip, tipIndex) => (
                              <li
                                key={tipIndex}
                                className="text-lightgray/80 text-xs flex items-start"
                              >
                                <span className="text-electricblue mr-2">
                                  •
                                </span>
                                {tip}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* SMALL CONTROLLED GAP */}
      <div className="h-8" />

      {/* Important Notes Section */}
      <div className="mt-16 bg-lightgray/5 rounded-3xl p-8 mx-6">
        <h2 className="text-2xl font-semibold text-lightpink mb-4 text-center">
          important notes
        </h2>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="text-electricblue font-medium mb-2">data privacy</h3>
            <p className="text-lightgray/80 text-sm">
              all uploaded audio files are processed anonymously and used only
              for educational and research purposes.{" "}
              <span className="font-bold text-lightpink">
                no personal information is stored.
              </span>
            </p>
          </div>
          <div>
            <h3 className="text-electricblue font-medium mb-2">best results</h3>
            <p className="text-lightgray/80 text-sm">
              for optimal analysis results, use clear,{" "}
              <span className="font-bold text-lightpink">monophonic</span>{" "}
              recordings without background noise. violin and voice work best
              with this tool.{" "}
              <span className="font-bold text-lightpink">polyphonic</span>{" "}
              recordings (multiple notes at once) are also supported, but with
              fewer analysis features available.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HowToUse;
