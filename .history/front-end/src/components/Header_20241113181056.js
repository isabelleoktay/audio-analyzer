import React from "react";

const Header = ({ title }) => {
  return (
    <header className="w-full bg-blue-500 py-3 text-center font-chakra">
      <h1 className="text-5xl text-white font-semibold tracking-widest">
        {title}
      </h1>
    </header>
  );
};
