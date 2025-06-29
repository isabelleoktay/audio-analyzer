import SecondaryButton from "../buttons/SecondaryButton";

const TestingCompleted = ({ subjectData }) => {
  const downloadSubjectData = () => {
    const processSubjectData = async (data) => {
      if (typeof data !== "object" || data === null) {
        return data; // Return non-object values as-is
      }

      if (data instanceof Blob) {
        // Convert Blob to Base64
        return await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(data);
        });
      }

      // Recursively process nested objects or arrays
      if (Array.isArray(data)) {
        return Promise.all(data.map((item) => processSubjectData(item)));
      }

      const processedData = {};
      for (const key in data) {
        processedData[key] = await processSubjectData(data[key]);
      }
      return processedData;
    };

    processSubjectData(subjectData).then((processedData) => {
      const dataStr = JSON.stringify(processedData, null, 2); // Convert processed data to JSON string
      const blob = new Blob([dataStr], { type: "application/json" }); // Create a Blob object
      const url = URL.createObjectURL(blob); // Create a URL for the Blob
      const link = document.createElement("a"); // Create a temporary anchor element
      link.href = url;
      link.download = `subject-data-${subjectData.subjectId || "unknown"}.json`; // Set the file name
      link.click(); // Trigger the download
      URL.revokeObjectURL(url); // Clean up the URL object
    });
  };

  const handleFeedback = () => {
    window.open("https://forms.gle/WF8g6WrMVsrokqyK6", "_blank");
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray w-1/2 space-y-8">
      <div className="text-4xl text-electricblue font-bold">
        Testing Completed.
      </div>
      <div className="text-lg text-justify">
        Thank you for participating in our audio feedback analysis testing tool!
        Your data has been successfully recorded and saved. If you would like to
        access your data, click the download subject data button.
      </div>
      <div className="flex flex-col items-center justify-center w-full space-y-4">
        <SecondaryButton onClick={() => (window.location.href = "/")}>
          go to audio analyzer
        </SecondaryButton>
        <SecondaryButton onClick={downloadSubjectData}>
          download subject data
        </SecondaryButton>
      </div>
      <div className="text-lg text-justify">
        {" "}
        Please fill out our feedback form to provide some more details on your
        experience with the audio analyzer.
      </div>
      <div className="flex items-center space-x-2">
        <SecondaryButton onClick={handleFeedback}>
          go to feedback form
        </SecondaryButton>
      </div>
    </div>
  );
};

export default TestingCompleted;
