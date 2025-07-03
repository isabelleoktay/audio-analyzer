// src/components/Layout.jsx
const Layout = ({ children }) => {
  return (
    <div className="px-6 md:px-12 lg:px-24 xl:px-32 max-w-screen-2xl mx-auto w-full flex-grow pb-20 min-h-screen">
      {children}
    </div>
  );
};

export default Layout;
