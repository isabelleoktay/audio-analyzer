// src/components/NavBar.jsx
import { useLocation } from "react-router-dom";
import ButtonGroup from "./buttons/ButtonGroup";
import SecondaryButton from "./buttons/SecondaryButton";

const NavBar = ({
  setSelectedInstrument,
  setUploadedFile,
  setInRecordMode,
  setAudioBlob,
  setAudioName,
  setAudioURL,
  setSelectedAnalysisFeature,
}) => {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const handleReset = () => {
    setSelectedInstrument(null);
    setUploadedFile(null);
    setInRecordMode(false);
    setAudioBlob(null);
    setAudioName("untitled.wav");
    setAudioURL(null);
    setSelectedAnalysisFeature(null);
  };

  const handleTooltips = () => {};

  const handleFeedback = () => {};

  return (
    <nav className="fixed top-0 left-0 w-full h-16 flex justify-between items-center px-6 py-4 bg-transparent text-white z-50">
      <div className="flex items-center space-x-2">
        <SecondaryButton onClick={handleReset}>reset</SecondaryButton>
        <SecondaryButton onClick={handleTooltips}>?</SecondaryButton>
      </div>

      <ButtonGroup
        buttons={[
          { type: "left", to: "/", label: "analyzer", active: isActive("/") },
          {
            type: "center",
            to: "/how-to-use",
            label: "how to use",
            active: isActive("/how-to-use"),
          },
          {
            type: "right",
            to: "https://github.com/your-repo",
            label: "github",
            external: true,
          },
        ]}
      />

      <div>
        <SecondaryButton onClick={handleFeedback}>feedback</SecondaryButton>
      </div>
    </nav>
  );
};

export default NavBar;
