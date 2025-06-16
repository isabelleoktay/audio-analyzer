// src/components/buttons/CenterButton.jsx
import { Link } from "react-router-dom";

const CenterButton = ({
  to,
  label,
  active = false,
  activeClassName = "text-blueblack bg-radial from-darkpink to-lightpink",
  inactiveClassName = "text-lightgray bg-lightgray bg-opacity-25 hover:bg-radial hover:from-darkpink/50 hover:to-lightpink/50",
  asButton = false,
  onClick,
}) => {
  const className = `px-4 py-2 text-sm font-medium ${
    active ? `${activeClassName}` : `${inactiveClassName}`
  }`;

  return asButton ? (
    <button className={className} onClick={onClick}>
      {label}
    </button>
  ) : (
    <Link to={to} className={className}>
      {label}
    </Link>
  );
};

export default CenterButton;
