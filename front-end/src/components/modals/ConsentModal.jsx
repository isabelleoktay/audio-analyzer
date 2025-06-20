import { useState } from "react";
import TertiaryButton from "../buttons/TertiaryButton";

const ConsentModal = ({ isOpen, onConsent }) => {
  const [isAgreed, setIsAgreed] = useState(false);

  const handleAgree = () => {
    if (isAgreed) {
      onConsent(true);
    }
  };

  const handleDisagree = () => {
    onConsent(false);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-blueblack bg-opacity-50" />

      {/* Modal */}
      <div className="relative bg-bluegray rounded-3xl p-8 max-w-2xl mx-4 shadow-2xl">
        <div className="text-center">
          {/* Header */}
          <h2 className="text-2xl font-bold text-lightpink mb-6">
            data collection consent
          </h2>

          {/* Content */}
          <div className="text-lightgray text-left space-y-4 mb-8">
            <p>
              by using this application, you consent to having any data
              collected used for educational and research purposes in the
              future.
            </p>

            <div className="bg-blueblack/50 p-4 rounded-lg border border-lightgray/10">
              <h3 className="text-lightpink font-semibold mb-2">
                data collection includes:
              </h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>audio recordings you upload or record</li>
                <li>generated audio features and analysis results</li>
              </ul>
            </div>

            <div className="bg-blueblack/50 p-4 rounded-lg border border-lightgray/10">
              <h3 className="text-lightpink font-semibold mb-2">
                privacy protection:
              </h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>all data collected is completely anonymous</li>
                <li>no personal identifying information is stored</li>
                <li>
                  data is used solely for research and educational purposes
                </li>
              </ul>
            </div>

            <p className="text-sm text-lightgray/70">
              this consent is required to use the application. your
              participation helps improve audio analysis research and education.
            </p>
          </div>

          {/* Checkbox */}
          <div className="flex items-center justify-center mb-6">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={isAgreed}
                onChange={(e) => setIsAgreed(e.target.checked)}
                className="w-4 h-4 rounded mr-3 accent-lightgray"
              />
              <span className="text-lightgray text-sm">
                i understand and agree to the data collection terms
              </span>
            </label>
          </div>

          {/* Buttons */}
          <div className="flex space-x-4 justify-center">
            <TertiaryButton onClick={handleDisagree}>disagree</TertiaryButton>
            <TertiaryButton
              onClick={handleAgree}
              active={isAgreed}
              disabled={!isAgreed}
            >
              agree & continue
            </TertiaryButton>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConsentModal;
