import FileUploadCard from "../cards/FileUploadCard";

const FileUploadSection = ({
  handleSwitchToRecordMode,
  handleFileUpload,
  inputFile,
  testingEnabled,
  className = "",
}) => {
  return (
    <div className={`flex flex-col ${className} w-full`}>
      <div className="flex flex-row justify-end align-bottom mb-1 w-full">
        <div
          onClick={handleSwitchToRecordMode}
          className="text-sm text-lightgray opacity-75 hover:opacity-100 cursor-pointer w-full text-right"
        >
          {testingEnabled ? "record audio" : "record audio instead"}
        </div>
      </div>
      <FileUploadCard
        onFileUpload={handleFileUpload}
        inputFile={inputFile}
        testingEnabled={testingEnabled}
      />
    </div>
  );
};

export default FileUploadSection;
