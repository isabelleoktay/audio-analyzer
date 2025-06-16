// src/components/buttons/RightButton.jsx
import { Link } from "react-router-dom";

const RightButton = ({
  to,
  label,
  active = false,
  external = false,
  asButton = false,
  onClick,
  activeClassName = "text-blueblack bg-radial from-darkpink to-lightpink",
  inactiveClassName = "text-lightgray bg-lightgray bg-opacity-25 hover:bg-radial hover:from-darkpink/50 hover:to-lightpink/50",
}) => {
  const className = `px-4 py-2 text-sm font-medium rounded-r-full ${
    active ? `${activeClassName}` : `${inactiveClassName}`
  }`;

  if (asButton) {
    return (
      <button className={className} onClick={onClick}>
        {label}
      </button>
    );
  }

  return external ? (
    <a
      href={to}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
    >
      {label}
    </a>
  ) : (
    <Link to={to} className={className}>
      {label}
    </Link>
  );
};

export default RightButton;
