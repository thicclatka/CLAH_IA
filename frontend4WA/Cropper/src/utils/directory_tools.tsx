import { ChangeEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { DirectoryState } from "../types";

export const fetchDirectoryData = (currentDir: string) => {
  return useQuery({
    queryKey: ["directory", currentDir],
    queryFn: async () => {
      const response = await fetch(
        `/api/list_directory/?directory=${encodeURIComponent(currentDir)}`
      );
      const data = await response.json();
      console.log("Directory data:", data); // Debug log
      return data;
    },
    enabled: !!currentDir,
  });
};

export const handleUpOneLevel = (
  state: DirectoryState,
  setState: (state: DirectoryState) => void,
  setSelectedFile: (file: string | null) => void
): void => {
  const parentDir =
    state.currentDir === "/"
      ? "/"
      : state.currentDir.substring(0, state.currentDir.lastIndexOf("/")) || "/";
  setState({ ...state, currentDir: parentDir });
  setSelectedFile(null);
};

export const handleDirectoryClick = (
  state: DirectoryState,
  setState: (state: DirectoryState) => void,
  dir: string
): void => {
  const newPath =
    state.currentDir === "/" ? `/${dir}` : `${state.currentDir}/${dir}`;
  setState({ ...state, currentDir: newPath });
};

export const handleInputChange = (
  state: DirectoryState,
  setState: (state: DirectoryState) => void,
  event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
): void => {
  setState({ ...state, newDir: event.target.value });
};

export const handleInputSubmit = (
  state: DirectoryState,
  setState: (state: DirectoryState) => void
): void => {
  if (state.newDir && state.newDir !== state.currentDir) {
    setState({ ...state, currentDir: state.newDir });
  }
};

export const handleFileClick = (
  state: DirectoryState,
  setSelectedFile: (file: string | null) => void,
  file: string
): void => {
  if (file.endsWith(".isxd")) {
    const fullPath =
      state.currentDir === "/" ? `/${file}` : `${state.currentDir}/${file}`;
    setSelectedFile(fullPath);
  }
};

export const loadDirectoryData = (currentDir: string) => {
  const {
    data: directoryData,
    isLoading,
    error,
  } = fetchDirectoryData(currentDir);

  return {
    directoryData,
    isLoading,
    error,
  };
};
