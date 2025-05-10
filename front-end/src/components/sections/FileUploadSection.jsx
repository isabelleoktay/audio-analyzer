import FileUploadCard from "../cards/FileUploadCard";

const FileUploadSection = ({
  handleSwitchToRecordMode,
  handleFileUpload,
  uploadedFile,
  className = "",
}) => {
  return (
    <div className={`flex flex-col ${className}`}>
      <div className="flex flex-row justify-end align-bottom mb-1">
        <div
          onClick={handleSwitchToRecordMode}
          className="text-sm text-lightgray opacity-75 hover:opacity-100 cursor-pointer"
        >
          record audio instead
        </div>
      </div>
      <FileUploadCard
        onFileUpload={handleFileUpload}
        uploadedFile={uploadedFile}
      />
    </div>
  );
};

export default FileUploadSection;
