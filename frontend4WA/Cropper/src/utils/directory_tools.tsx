import { FC, ChangeEvent, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { DirectoryResponse, DirectoryState } from "../types";
import FileViewer from "../components/FileViewer";
// import "../styles/directory_tools.css";

const DirectoryNavigation: FC = () => {
  const [state, setState] = useState<DirectoryState>({
    currentDir: "/",
    directories: [],
    files: [],
    newDir: "",
  });
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const {
    data: directoryData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["directory", state.currentDir],
    queryFn: () =>
      fetch(
        `/api/list_directory/?directory=${encodeURIComponent(state.currentDir)}`
      ).then((res) => res.json()),
    enabled: !!state.currentDir,
  });

  const handleUpOneLevel = (): void => {
    const parentDir =
      state.currentDir === "/"
        ? "/"
        : state.currentDir.substring(0, state.currentDir.lastIndexOf("/")) ||
          "/";
    setState((prev: DirectoryState) => ({ ...prev, currentDir: parentDir }));
  };

  const handleDirectoryClick = (dir: string): void => {
    const newPath =
      state.currentDir === "/" ? `/${dir}` : `${state.currentDir}/${dir}`;
    setState((prev: DirectoryState) => ({ ...prev, currentDir: newPath }));
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    setState((prev: DirectoryState) => ({
      ...prev,
      newDir: event.target.value,
    }));
  };

  const handleInputSubmit = (): void => {
    if (state.newDir && state.newDir !== state.currentDir) {
      setState((prev: DirectoryState) => ({
        ...prev,
        currentDir: state.newDir,
      }));
    }
  };

  const handleFileClick = (file: string) => {
    if (file.endsWith(".isxd")) {
      const fullPath =
        state.currentDir === "/" ? `/${file}` : `${state.currentDir}/${file}`;
      setSelectedFile(fullPath);
    }
  };

  if (isLoading) {
    return <div>Loading directory contents...</div>;
  }

  if (error) {
    return <div>Error loading directory contents</div>;
  }

  return (
    <div className="directory-navigation">
      <div className="directory-header">
        <h2>Current Directory: {state.currentDir}</h2>
        <button onClick={handleUpOneLevel} className="nav-button">
          ⬆️ Up one level
        </button>
      </div>

      <div className="directory-input">
        <input
          type="text"
          value={state.newDir}
          onChange={handleInputChange}
          placeholder="Enter directory path"
          className="path-input"
        />
        <button onClick={handleInputSubmit} className="go-button">
          Go
        </button>
      </div>

      <div className="directory-lists">
        <div className="directories-section">
          <h3>Directories</h3>
          <ul className="directory-list">
            {directoryData?.directories.map((dir: string) => (
              <li
                key={dir}
                onClick={() => handleDirectoryClick(dir)}
                className="directory-item"
              >
                📁 {dir}
              </li>
            ))}
          </ul>
        </div>

        <div className="files-section">
          <h3>Files</h3>
          <ul className="file-list">
            {directoryData?.files.map((file: string) => (
              <li
                key={file}
                onClick={() => handleFileClick(file)}
                className={`file-item ${
                  file.endsWith(".isxd") ? "isxd-file" : ""
                }`}
              >
                {file.endsWith(".isxd") ? "🎥" : "📄"} {file}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {selectedFile && (
        <div className="file-viewer-container">
          <h3>Selected File: {selectedFile.split("/").pop()}</h3>
          <FileViewer filePath={selectedFile} />
        </div>
      )}
    </div>
  );
};

export default DirectoryNavigation;
