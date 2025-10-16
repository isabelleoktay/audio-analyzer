import UploadAudioCard from "../components/cards/UploadAudioCard";

const MultiAudio = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8">
      <UploadAudioCard label="Reference Audio" />
    </div>
  );
};

export default MultiAudio;
