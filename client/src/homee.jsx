import React from 'react';

const LegalEaseSection = () => {
  return (
    <div className="bg-gray-100 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:text-center">
          <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            How does LegalEase work?
          </h2>
        </div>
        <div className="mt-10">
          <div className="flex flex-col items-center md:flex-row md:justify-between">
            <div className="md:w-1/2 lg:w-5/12">
              <h3 className="text-2xl font-extrabold text-gray-900">
                Guidance for new users
              </h3>
              <p className="mt-4 text-lg text-gray-500">
                New users can easily access legal assistance on LegalEase
                without prior legal knowledge. Simply sign up and get matched
                with a legal expert to address your concerns!
              </p>
              <button className="mt-8 inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700">
                Discover more
              </button>
            </div>
            <div className="mt-12 md:mt-0 md:w-1/2 lg:w-6/12">
              <img
                src="logo.png"
                alt="Illustration"
                className="mx-auto"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LegalEaseSection;