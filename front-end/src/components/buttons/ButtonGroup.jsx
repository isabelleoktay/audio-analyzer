// src/components/buttons/ButtonGroup.jsx
import LeftButton from "./LeftButton";
import CenterButton from "./CenterButton";
import RightButton from "./RightButton";
const ButtonGroup = ({ buttons, className = "" }) => {
  /**
   * buttons: array of { type: 'left' | 'center' | 'right', label: string, to: string, active?: boolean, external?: boolean }
   */
  return (
    <div className={`flex ${className}`}>
      {buttons.map((btn, idx) => {
        switch (btn.type) {
          case "left":
            return <LeftButton key={idx} {...btn} />;
          case "center":
            return <CenterButton key={idx} {...btn} />;
          case "right":
            return <RightButton key={idx} {...btn} />;
          default:
            return null;
        }
      })}
    </div>
  );
};

export default ButtonGroup;
