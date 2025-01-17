import React from "react";

const Header = ({ title }) => {
  return (
    <header className="w-full bg-blue-500 py-3 text-center font-chakra">
      <div className="flex flex-col">
        <h1 className="text-5xl text-white font-semibold tracking-widest mb-1">
          {title}
        </h1>
        <a
          className="text-blue-200 hover:text-opacity-75 tracking-widest font-poppins"
          href="https://forms.gle/aSRtqaxEMot2HCvH7"
          target="_blank"
        >
          Click here to send feedback!
        </a>
      </div>
    </header>
  );
};

export default Header;
