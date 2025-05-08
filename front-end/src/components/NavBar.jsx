// src/components/NavBar.jsx
import { useLocation } from "react-router-dom";
import ButtonGroup from "./buttons/ButtonGroup";
import SecondaryButton from "./buttons/SecondaryButton";

export default function NavBar() {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const handleReset = () => {};

  const handleTooltips = () => {};

  const handleFeedback = () => {};

  return (
    <nav className="flex justify-between items-center px-6 py-4 bg-transparent text-white">
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
}
