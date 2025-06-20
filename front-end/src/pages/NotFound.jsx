import { useNavigate } from "react-router-dom";
import SecondaryButton from "../components/buttons/SecondaryButton";

const NotFound = () => {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8">
      <div className="flex flex-col items-center space-y-8 max-w-md text-center">
        {/* 404 Text */}
        <div className="space-y-2">
          <h1 className="text-9xl font-bold text-lightpink">404</h1>
          <h2 className="text-2xl font-semibold text-lightpink">
            Oops! Page Not Found
          </h2>
        </div>

        {/* Navigation buttons */}
        <div className="flex flex-col sm:flex-row gap-4">
          <SecondaryButton onClick={() => navigate("/")}>
            return to analyzer
          </SecondaryButton>
          <SecondaryButton onClick={() => navigate(-1)}>
            go back
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
