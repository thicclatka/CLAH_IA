import { FC, useState } from "react";
import { useQuery } from "@tanstack/react-query";
// import "../styles/FileViewer.css";

interface FileViewerProps {
  filePath: string;
}

interface FrameData {
  totalFrames: number;
  currentFrame: number;
  frameData?: string; // Base64 encoded image data
}

const FileViewer: FC<FileViewerProps> = ({ filePath }) => {
  const [currentFrame, setCurrentFrame] = useState<number>(0);

  // Query for total frames
  const { data: totalFramesData, isLoading: isLoadingTotalFrames } = useQuery({
    queryKey: ["totalFrames", filePath],
    queryFn: () =>
      fetch(`/api/load_isxd/?file_path=${encodeURIComponent(filePath)}`).then(
        (res) => res.json()
      ),
    enabled: !!filePath,
  });

  // Query for current frame
  const { data: frameData, isLoading: isLoadingFrame } = useQuery({
    queryKey: ["frame", filePath, currentFrame],
    queryFn: async () => {
      const response = await fetch(
        `/api/get_frame/?file_path=${encodeURIComponent(
          filePath
        )}&frame_idx=${currentFrame}`
      );
      if (!response.ok) throw new Error("Failed to fetch frame");
      const blob = await response.blob();
      return URL.createObjectURL(blob);
    },
    enabled:
      !!filePath &&
      currentFrame >= 0 &&
      currentFrame < (totalFramesData?.total_frames || 0),
  });

  const handlePrevFrame = () => {
    if (currentFrame > 0) {
      setCurrentFrame((prev) => prev - 1);
    }
  };

  const handleNextFrame = () => {
    if (currentFrame < (totalFramesData?.total_frames || 0) - 1) {
      setCurrentFrame((prev) => prev + 1);
    }
  };

  if (isLoadingTotalFrames || isLoadingFrame) {
    return <div className="file-viewer loading">Loading...</div>;
  }

  if (!totalFramesData) {
    return <div className="file-viewer error">Failed to load file</div>;
  }

  return (
    <div className="file-viewer">
      <div className="frame-navigation">
        <button
          onClick={handlePrevFrame}
          disabled={currentFrame === 0}
          className="nav-button"
        >
          Previous Frame
        </button>
        <span className="frame-counter">
          Frame {currentFrame + 1} of {totalFramesData.total_frames}
        </span>
        <button
          onClick={handleNextFrame}
          disabled={currentFrame === totalFramesData.total_frames - 1}
          className="nav-button"
        >
          Next Frame
        </button>
      </div>

      {frameData && (
        <div className="frame-container">
          <img
            src={frameData}
            alt={`Frame ${currentFrame + 1}`}
            className="frame-image"
          />
        </div>
      )}
    </div>
  );
};

export default FileViewer;
