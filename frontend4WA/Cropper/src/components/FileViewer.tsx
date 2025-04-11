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
  Paper,
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

  // Initialize crop box with existing coordinates if they exist
  useEffect(() => {
    if (cropCoords && cropCoords.length === 2) {
      const [start, end] = cropCoords;
      setCropBox({
        x: start[0],
        y: start[1],
        width: end[0] - start[0],
        height: end[1] - start[1],
      });
    }
  }, [cropCoords]);

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

    const rect = imageRef.current.getBoundingClientRect();
    // Clamp the coordinates to the image boundaries before exporting
    const x1 = Math.max(0, Math.min(cropBox.x, rect.width));
    const y1 = Math.max(0, Math.min(cropBox.y, rect.height));
    const x2 = Math.max(0, Math.min(cropBox.x + cropBox.width, rect.width));
    const y2 = Math.max(0, Math.min(cropBox.y + cropBox.height, rect.height));

    try {
      const response = await fetch(
        `/api/export_crop_coords/?file_path=${encodeURIComponent(filePath)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            x1: Math.round(x1),
            y1: Math.round(y1),
            x2: Math.round(x2),
            y2: Math.round(y2),
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error response:", errorData);
        throw new Error(JSON.stringify(errorData, null, 2));
      }

      alert("Crop coordinates exported successfully!");
    } catch (error: any) {
      console.error("Error exporting crop coordinates:", error);
      alert("Failed to export crop coordinates:\n" + error.message);
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
