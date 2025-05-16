import { FC, useState, useEffect, useCallback, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  CardMedia,
  Box,
  Stack,
  Typography,
  CircularProgress,
  IconButton,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import CropIcon from "@mui/icons-material/Crop";
import SaveIcon from "@mui/icons-material/Save";
import { CropBox, MovieData, FileViewerProps } from "../types";

const FRAME_RATE = 5; // frames per second
const FRAME_INTERVAL = 1000 / FRAME_RATE; // milliseconds per frame

const FileViewer: FC<FileViewerProps> = ({ filePath, cropCoords }) => {
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [movieData, setMovieData] = useState<MovieData | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isCropping, setIsCropping] = useState<boolean>(false);
  const [cropBox, setCropBox] = useState<CropBox | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const imageRef = useRef<HTMLImageElement>(null);

  // Query for movie data (frames and total frames)
  const { isLoading } = useQuery({
    queryKey: ["movie", filePath],
    queryFn: async () => {
      const response = await fetch(
        `/api/import_movie/?file_path=${encodeURIComponent(filePath)}`
      );
      if (!response.ok) throw new Error("Failed to fetch movie");
      const data = await response.json();
      setMovieData(data);
      return data;
    },
    enabled: !!filePath,
  });

  useEffect(() => {
    if (
      cropCoords &&
      cropCoords.length === 2 &&
      imageRef.current &&
      imageRef.current.naturalWidth > 0 &&
      imageRef.current.naturalHeight > 0
    ) {
      const [start, end] = cropCoords; // original image coordinates

      const rect = imageRef.current.getBoundingClientRect(); // Displayed size
      const naturalWidth = imageRef.current.naturalWidth;
      const naturalHeight = imageRef.current.naturalHeight;

      if (rect.width === 0 || rect.height === 0) {
        return;
      }

      const scaleX = rect.width / naturalWidth; // display_width / original_width
      const scaleY = rect.height / naturalHeight; // display_height / original_height

      const display_x1 = start[0] * scaleX;
      const display_y1 = start[1] * scaleY;
      const display_x2 = end[0] * scaleX;
      const display_y2 = end[1] * scaleY;

      setCropBox({
        x: Math.min(display_x1, display_x2),
        y: Math.min(display_y1, display_y2),
        width: Math.abs(display_x2 - display_x1),
        height: Math.abs(display_y2 - display_y1),
      });
    } else if (!cropCoords) {
      // If cropCoords are empty, clear the box
      setCropBox(null);
    }
    // Adding dependencies on imageRef.current properties that affect scaling
    // Note: directly depending on imageRef.current might not trigger re-renders if only its properties change.
    // A common pattern is to use a state variable that updates on image load or resize.
    // However, for initial load, naturalWidth/Height check often suffices if image is loaded before this effect runs.
    // If the image resizes dynamically, a ResizeObserver on imageRef might be more robust for updating the display cropBox.
  }, [
    cropCoords,
    imageRef.current?.naturalWidth,
    imageRef.current?.naturalHeight,
    imageRef.current?.width,
    imageRef.current?.height,
  ]);

  const handlePrevFrame = useCallback(() => {
    if (currentFrame > 0) {
      setCurrentFrame((prev) => prev - 1);
    }
  }, [currentFrame]);

  const handleNextFrame = useCallback(() => {
    if (movieData && currentFrame < movieData.total_frames - 1) {
      setCurrentFrame((prev) => prev + 1);
    }
  }, [currentFrame, movieData]);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (isPlaying && movieData) {
      intervalId = setInterval(() => {
        setCurrentFrame((prev) => {
          if (prev >= movieData.total_frames - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, FRAME_INTERVAL);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isPlaying, movieData]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isCropping || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setStartPos({ x, y });
    setIsDragging(true);
    setCropBox({ x, y, width: 0, height: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !startPos || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Calculate the crop box dimensions
    const width = x - startPos.x;
    const height = y - startPos.y;

    // Calculate the new position and dimensions
    const newX = width < 0 ? x : startPos.x;
    const newY = height < 0 ? y : startPos.y;
    const newWidth = Math.abs(width);
    const newHeight = Math.abs(height);

    // Ensure the crop box stays within the image boundaries
    const clampedX = Math.max(0, Math.min(newX, rect.width - newWidth));
    const clampedY = Math.max(0, Math.min(newY, rect.height - newHeight));

    setCropBox({
      x: clampedX,
      y: clampedY,
      width: newWidth,
      height: newHeight,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setStartPos(null);
  };

  const handleExportCrop = async () => {
    if (!cropBox || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect(); // Displayed size
    const naturalWidth = imageRef.current.naturalWidth; // Original (natural) width of the image source
    const naturalHeight = imageRef.current.naturalHeight; // Original (natural) height of the image source

    // Prevent division by zero if image dimensions are not loaded yet
    if (
      rect.width === 0 ||
      rect.height === 0 ||
      naturalWidth === 0 ||
      naturalHeight === 0
    ) {
      alert("Error: Image dimensions are not fully loaded. Please try again.");
      return;
    }

    // Calculate scaling factors
    const scaleX = naturalWidth / rect.width;
    const scaleY = naturalHeight / rect.height;

    // Scale the cropBox coordinates (which are relative to the displayed image)
    // back to the original image's coordinate system.
    const original_x1 = cropBox.x * scaleX;
    const original_y1 = cropBox.y * scaleY;
    const original_x2 = (cropBox.x + cropBox.width) * scaleX;
    const original_y2 = (cropBox.y + cropBox.height) * scaleY;

    // Prepare data to send to the backend
    // Ensure x1,y1 is top-left and x2,y2 is bottom-right for consistency,
    // though the backend's use of min/max for LRTB handles arbitrary corners.
    const dataToSend = {
      x1: Math.round(Math.min(original_x1, original_x2)),
      y1: Math.round(Math.min(original_y1, original_y2)),
      x2: Math.round(Math.max(original_x1, original_x2)),
      y2: Math.round(Math.max(original_y1, original_y2)),
    };

    try {
      const response = await fetch(
        `/api/export_crop_coords/?file_path=${encodeURIComponent(filePath)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(dataToSend),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error response:", errorData);
        throw new Error(JSON.stringify(errorData, null, 2));
      }

      alert("Crop coordinates exported successfully!");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      console.error("Error exporting crop coordinates:", error);
      alert("Failed to export crop coordinates:\n" + errorMessage);
    }
  };

  if (isLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="200px"
      >
        <Stack spacing={2} alignItems="center">
          <CircularProgress />
          <Typography variant="body1">Loading frames...</Typography>
        </Stack>
      </Box>
    );
  }

  if (!movieData) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="200px"
      >
        <Typography color="error">Failed to load frames</Typography>
      </Box>
    );
  }

  return (
    <Box className="file-viewer">
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="center"
        sx={{ mb: 2 }}
      >
        <Button
          variant="contained"
          onClick={handlePrevFrame}
          disabled={currentFrame === 0 || isPlaying}
        >
          Previous
        </Button>
        <IconButton onClick={togglePlay} color="primary" size="large">
          {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
        </IconButton>
        <Button
          variant="contained"
          onClick={handleNextFrame}
          disabled={currentFrame === movieData.total_frames - 1 || isPlaying}
        >
          Next
        </Button>
        <Typography variant="body1" sx={{ mx: 2 }}>
          Frame {currentFrame + 1} of {movieData.total_frames}
        </Typography>
        <IconButton
          onClick={() => setIsCropping(!isCropping)}
          color={isCropping ? "secondary" : "default"}
        >
          <CropIcon />
        </IconButton>
        {cropBox && (
          <IconButton onClick={handleExportCrop} color="primary">
            <SaveIcon />
          </IconButton>
        )}
      </Stack>

      {movieData.frames[currentFrame] && (
        <Box
          sx={{
            maxWidth: "800px",
            margin: "20px auto",
            display: "flex",
            justifyContent: "center",
            position: "relative",
            userSelect: "none",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <CardMedia
            component="img"
            src={`data:image/jpeg;base64,${movieData.frames[currentFrame]}`}
            alt={`Frame ${currentFrame + 1}`}
            ref={imageRef}
            draggable={false}
            onDragStart={(e) => e.preventDefault()}
            sx={{
              maxHeight: "500px",
              width: "auto",
              objectFit: "contain",
              cursor: isCropping ? "crosshair" : "default",
              userSelect: "none",
              WebkitUserDrag: "none",
              MozUserSelect: "none",
              msUserSelect: "none",
            }}
          />
          {cropBox && cropCoords && (
            <Box
              sx={{
                position: "absolute",
                border: "2px solid green",
                backgroundColor: "rgba(0, 255, 0, 0.01)",
                left: `${cropBox.x}px`,
                top: `${cropBox.y}px`,
                width: `${cropBox.width}px`,
                height: `${cropBox.height}px`,
                pointerEvents: "none",
              }}
            />
          )}
        </Box>
      )}
    </Box>
  );
};

export default FileViewer;
