import { FC, useState } from "react";
import { useTheme } from "@mui/material/styles";
import {
  Box,
  Toolbar,
  Typography,
  Button,
  Paper,
  IconButton,
  Drawer,
  ListItem,
  ListItemText,
  Divider,
  TextField,
} from "@mui/material";
import Brightness7Icon from "@mui/icons-material/Brightness7";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import { helpText } from "../components/help";
import {
  RightColumn,
  LeftColumn,
  DirectoryListContainer,
  DirectoryHeader,
  DirectoryInput,
  Section,
  ListContainer,
  StyledListItem,
  DirectoryContainer,
  FileViewerContainer,
} from "./layout_const";
import { DirectoryState, DirectoryResponse } from "../types";
import {
  handleUpOneLevel,
  handleDirectoryClick,
  handleInputChange,
  handleInputSubmit,
  handleFileClick,
  loadDirectoryData,
} from "../utils/directory_tools";
import FileViewer from "../components/FileViewer";

interface LayoutProps {
  onToggleColorMode: () => void;
}

const Header = ({ onToggleColorMode }: { onToggleColorMode: () => void }) => {
  const theme = useTheme();
  const [helpOpen, setHelpOpen] = useState(false);

  return (
    <Paper
      sx={{
        p: 2,
        mb: 1,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <img
            src="/docs/images/png/Cropper_icon.png"
            alt="Cropping Icon"
            style={{ width: "70px", height: "70px" }}
          />
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              1P Cropping Utility
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Export cropped dimensions for 1P experiments
            </Typography>
          </Box>
        </Box>
      </Box>
      <Toolbar>
        <IconButton onClick={() => setHelpOpen(true)} color="inherit">
          <HelpOutlineIcon />
        </IconButton>
        <IconButton onClick={onToggleColorMode} color="inherit">
          {theme.palette.mode === "dark" ? (
            <Brightness7Icon />
          ) : (
            <Brightness4Icon />
          )}
        </IconButton>
      </Toolbar>

      <Drawer
        anchor="right"
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
        sx={{
          width: 400,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: 400,
            boxSizing: "border-box",
            padding: 2,
          },
        }}
      >
        <Typography variant="h6" sx={{ mb: 2 }}>
          Help
        </Typography>
        <Divider />
        <Box sx={{ mt: 2 }}>
          {helpText.split("\n").map((line, index) => (
            <Typography key={index} sx={{ mb: 1, whiteSpace: "pre-line" }}>
              {line}
            </Typography>
          ))}
        </Box>
      </Drawer>
    </Paper>
  );
};

const DirectoryLayout: FC<{
  state: DirectoryState;
  setState: (state: DirectoryState) => void;
  setSelectedFile: (file: string | null) => void;
  directoryData: DirectoryResponse;
  isLoading: boolean;
  error: Error | null;
}> = ({
  state,
  setState,
  setSelectedFile,
  directoryData,
  isLoading,
  error,
}) => {
  if (isLoading) {
    return <Typography>Loading directory contents...</Typography>;
  }

  if (error) {
    return (
      <Typography color="error">Error loading directory contents</Typography>
    );
  }
  return (
    <Box>
      <DirectoryHeader>
        <Typography variant="subtitle2" color="text.secondary">
          Current Directory:
        </Typography>
        <Typography variant="body2" noWrap>
          {state.currentDir}
        </Typography>
        <Button
          variant="contained"
          onClick={() => handleUpOneLevel(state, setState, setSelectedFile)}
          startIcon="⬆️"
          fullWidth
        >
          Up one level
        </Button>
      </DirectoryHeader>

      <DirectoryInput>
        <TextField
          fullWidth
          value={state.newDir}
          onChange={(e) => handleInputChange(state, setState, e)}
          placeholder="Enter directory path"
          variant="outlined"
          size="small"
        />
        <Button
          variant="contained"
          onClick={() => handleInputSubmit(state, setState)}
          fullWidth
        >
          Go
        </Button>
      </DirectoryInput>

      <Section>
        <Typography variant="subtitle1" gutterBottom>
          Directories
        </Typography>
        <DirectoryListContainer>
          {directoryData?.directories.map((dir: string) => (
            <StyledListItem
              key={dir}
              onClick={() => handleDirectoryClick(state, setState, dir)}
            >
              <ListItemText primary={`📁 ${dir}`} />
            </StyledListItem>
          ))}
        </DirectoryListContainer>
      </Section>

      <Divider sx={{ my: 2 }} />

      <Section>
        <Typography variant="subtitle2" gutterBottom>
          Files
        </Typography>
        <DirectoryListContainer>
          {directoryData?.files.map((file: string) => (
            <StyledListItem
              key={file}
              onClick={() => handleFileClick(state, setSelectedFile, file)}
            >
              <ListItemText
                primary={`${file.endsWith(".isxd") ? "🎥" : "📄"} ${file}`}
                primaryTypographyProps={{
                  color: file.endsWith(".isxd") ? "primary" : "textPrimary",
                  fontWeight: file.endsWith(".isxd") ? "bold" : "normal",
                }}
              />
            </StyledListItem>
          ))}
        </DirectoryListContainer>
      </Section>

      <Divider sx={{ my: 2 }} />

      <Section>
        <Typography variant="subtitle2" gutterBottom>
          Crop Dimensions
        </Typography>
        <ListContainer>
          {directoryData?.crop_dim_coords &&
          Array.isArray(directoryData.crop_dim_coords) &&
          directoryData.crop_dim_coords.length > 0 ? (
            <ListItem sx={{ cursor: "default" }}>
              <ListItemText
                primary={`📏 ${JSON.stringify(directoryData.crop_dim_coords)}`}
                primaryTypographyProps={{
                  color: "textSecondary",
                }}
              />
            </ListItem>
          ) : (
            <ListItem sx={{ cursor: "default" }}>
              <ListItemText
                primary="No crop dimensions"
                primaryTypographyProps={{
                  color: "error",
                }}
              />
            </ListItem>
          )}
        </ListContainer>
      </Section>
    </Box>
  );
};

const FileViewerLayout: FC<{
  selectedFile: string | null;
  directoryData: DirectoryResponse;
}> = ({ selectedFile, directoryData }) => {
  return (
    <Box>
      {selectedFile && (
        <FileViewerContainer>
          <Typography variant="h6" gutterBottom>
            Selected File: {selectedFile.split("/").pop()}
          </Typography>
          <FileViewer
            filePath={selectedFile}
            cropCoords={directoryData?.crop_dim_coords}
          />
        </FileViewerContainer>
      )}
    </Box>
  );
};
export const Layout: FC<LayoutProps> = ({ onToggleColorMode }) => {
  const [state, setState] = useState<DirectoryState>({
    currentDir: "/",
    directories: [],
    files: [],
    crop_dim_coords: [],
    newDir: "",
  });
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const { directoryData, isLoading, error } = loadDirectoryData(
    state.currentDir
  );

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
      }}
    >
      <Header onToggleColorMode={onToggleColorMode} />

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 1,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 1,
            borderRadius: 2,
          }}
        >
          <DirectoryContainer>
            <LeftColumn>
              <DirectoryLayout
                state={state}
                setState={setState}
                setSelectedFile={setSelectedFile}
                directoryData={directoryData}
                isLoading={isLoading}
                error={error}
              />
            </LeftColumn>
            <RightColumn>
              <FileViewerLayout
                selectedFile={selectedFile}
                directoryData={directoryData}
              />
            </RightColumn>
          </DirectoryContainer>
        </Paper>
      </Box>
    </Box>
  );
};
