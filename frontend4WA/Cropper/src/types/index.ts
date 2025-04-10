export interface DirectoryEntry {
  name: string;
  isDirectory: boolean;
}

export interface DirectoryResponse {
  directories: string[];
  files: string[];
}

export interface DirectoryState {
  currentDir: string;
  directories: string[];
  files: string[];
  newDir: string;
  error?: string;
}
