const NotFound = () => {
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
      </div>
    </div>
  );
};

export default NotFound;
