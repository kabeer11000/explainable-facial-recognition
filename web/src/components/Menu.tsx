import { $ConnectionStatus } from '@/stores/connection';
import { useStore } from '@nanostores/react';
import { Fragment } from 'react';

const Menu = () => {
  const {connected, statusMessage} = useStore($ConnectionStatus);
  return (
    <>
      <div tabIndex={-1} className="flex z-101 bottom-0 right-0 gap-x-4 w-full absolute px-4 py-6 group-focus:opacity-0 md:lg:group-focus:opacity-100 duration-300 transition-opacity">
        <div>
          <div className="mb-4 text-white" style={{ mixBlendMode: "exclusion" }}>
            <p className="text-lg">Explainable Facial Recognition</p>
            <p className="text-sm max-w-lg text-neutral-200 hidden-md:lg:block">
              AI-Powered Video Analysis: Live processing of your video for real-time
              object detection and feature extraction.
            </p>
            <hr className='border-b-none my-4 border-neutral-500'/>
          </div>
        </div>
      </div>
      <div className="flex z-101 bottom-0 left-0 gap-x-4 text-white w-full justify-between absolute px-4 py-2">
        <div>
          <div className="flex items-center justify-start">
            {connected ? (
              <Fragment>
                <p className="text-md text-green-600">⬤</p>
                <p
                  className="text-sm font-semibold ml-2 cursor-pointer hover:scale-105 duration-100"
                  title="Cannot connect to the backend."
                >
                  Connected
                </p>

              </Fragment>
            ) : (
              <Fragment>
                <p className="text-md text-red-600">⬤</p>
                <p
                  className="text-md font-semibold ml-2 cursor-pointer hover:scale-105 duration-100"
                  title="Cannot connect to the backend.">
                  Disconnected
                </p>
              </Fragment>
            )}

          </div>
        </div>
        <div className='gap-x-4 flex justify-end'><a href="https://kabeers.network/research/">About</a><a href="/paper/document.pdf">Paper</a>
        <a href="/upload">Upload</a></div>

      </div>
    </>

  );
};

export default Menu;
