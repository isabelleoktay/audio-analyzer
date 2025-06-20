import React from "react";
import {
  FaMusic,
  FaUpload,
  FaMicrophone,
  FaChartLine,
  FaPlay,
} from "react-icons/fa";

const HowToUse = () => {
  const steps = [
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

  return (
    <div className="min-h-screen">
      <div className="max-w-4xl mx-auto my-24">
        {/* Steps */}
        <div className="space-y-6">
          {steps.map((step, index) => (
            <div key={index} className="bg-lightgray/10 rounded-3xl p-8">
              <div className="flex items-start space-x-6">
                {/* Icon */}
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 bg-gradient-to-r from-darkpink to-electricblue rounded-full flex items-center justify-center">
                    <step.icon className="text-2xl text-blueblack" />
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1">
                  <h4 className="text-xl font-semibold text-lightpink mb-3">
                    {step.title}
                  </h4>
                  <p className="text-lightgray mb-4">{step.description}</p>

                  {/* Tips */}
                  <div className="bg-blueblack/30 rounded-lg p-4">
                    <ul className="space-y-1">
                      {step.tips.map((tip, tipIndex) => (
                        <li
                          key={tipIndex}
                          className="text-lightgray/80 text-sm flex items-start"
                        >
                          <span className="text-electricblue mr-2">•</span>
                          {tip}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Additional Info */}
        <div className="mt-16 bg-lightgray/5 rounded-3xl p-8">
          <h2 className="text-2xl font-semibold text-lightpink mb-4 text-center">
            important notes
          </h2>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h3 className="text-electricblue font-medium mb-2">
                data privacy
              </h3>
              <p className="text-lightgray/80 text-sm">
                all uploaded audio files are processed anonymously and used only
                for educational and research purposes.{" "}
                <span className="font-bold text-lightpink">
                  no personal information is stored.
                </span>
              </p>
            </div>
            <div>
              <h3 className="text-electricblue font-medium mb-2">
                best results
              </h3>
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
    </div>
  );
};

export default HowToUse;
