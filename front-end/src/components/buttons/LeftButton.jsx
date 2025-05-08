// src/components/buttons/LeftButton.jsx
import { Link } from "react-router-dom";

const LeftButton = ({ to, label, active = false }) => {
  return (
    <Link
      to={to}
      className={`px-4 py-2 text-sm font-medium rounded-l-full ${
        active
          ? "text-blueblack bg-radial from-darkpink to-lightpink"
          : "text-lightgray bg-lightgray bg-opacity-25 hover:bg-radial hover:from-darkpink/50 hover:to-lightpink/50"
      }`}
    >
      {label}
    </Link>
  );
};

export default LeftButton;
