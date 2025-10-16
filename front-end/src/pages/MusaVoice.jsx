import React from "react";

const MusaVoice = () => {

  return (
    <div className="min-h-screen">
      <div className="max-w-4xl mx-auto my-24">
        {/* Welcome text for voice survey */}
        <h2 className="text-2xl font-semibold text-lightpink mb-4 text-center">
        We would like to know a little about your background
        </h2>
        <p className="text-lightgray/80 text-sm">
            Please answer the following questions to begin using the MuSA testing platform. These will provide us with an understanding of your vocal background and calibrate some aspects of the platform to your needs. 
        </p>
        <div className="mt-16 bg-lightgray/5 rounded-3xl p-8">
            <h2 className="text-2xl font-semibold text-lightpink mb-4 text-center">
                Question 1
            </h2>
        </div>
      </div>

    </div>
  );
};

export default MusaVoice;
