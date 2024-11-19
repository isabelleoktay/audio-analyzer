import React from "react";

const Tabs = ({ activeTab, setActiveTab, tabs }) => {
  return (
    <div className="flex flex-col justify-start items-center w-8">
      {tabs.map((tab, index) => (
        <button
          key={index}
          className={`py-2 px-1 flex-grow text-lg text-white font-semibold border-none bg-${
            tab.color
          } rounded-r-lg ${
            activeTab === tab.name
              ? ""
              : "transition ease-in-out delay-50 bg-opacity-50 hover:bg-opacity-75"
          }`}
          style={{
            writingMode: "vertical-rl",
            transform: "rotate(180deg)",
          }}
          onClick={() => setActiveTab(tab.name)}
        >
          {tab.name}
        </button>
      ))}
    </div>
  );
};

export default Tabs;
