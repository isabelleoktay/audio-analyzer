import React from "react";

const SmallTextLink = ({ nonLinkText = "", linkText, handleClick }) => {
  return (
    <div className="text-gray-500 text-sm my-2">
      {nonLinkText}
      <span
        className="underline hover:cursor-pointer hover:text-blue-500"
        onClick={handleClick}
      >
        {linkText}
      </span>
    </div>
  );
};

export default SmallTextLink;
