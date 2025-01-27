import React from "react";

const Header = ({ title }) => {
  return (
    <header className="w-full bg-blue-500 py-3 text-center font-chakra">
      <div className="flex flex-col">
        <h1 className="text-5xl text-white font-semibold tracking-widest mb-1">
          {title}
        </h1>
        <a
          className="text-yellow-400 text-2xl font-bold hover:scale-105 transform transition-all font-poppins inline-block"
          href="https://forms.gle/aSRtqaxEMot2HCvH7"
          target="_blank"
        >
          Click here to send us feedback!
        </a>
      </div>
    </header>
  );
};

export default Header;
