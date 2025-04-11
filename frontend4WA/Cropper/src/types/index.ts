export interface DirectoryEntry {
  name: string;
  isDirectory: boolean;
}

export interface DirectoryState {
  currentDir: string;
  directories: string[];
  files: string[];
  crop_dim_coords: number[][];
  newDir: string;
}

export interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FileViewerProps {
  filePath: string;
  cropCoords?: number[][];
}

export interface MovieData {
  total_frames: number;
  frames: string[];
}

export interface DirectoryResponse {
  directories: string[];
  files: string[];
  crop_dim_coords: number[][];
}
