import React from "react";

const MusaVoice = () => {

  return (
    <div className="min-h-screen">
      <div className="max-w-4xl mx-auto my-24">
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

export default MusaVoice;
