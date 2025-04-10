// Layout component for the application

import { FC } from "react";
import DirectoryNavigation from "../utils/directory_tools";
// import "../styles/layout.css";
import { useTheme } from "@mui/material/styles";
import { Box, AppBar, Toolbar, Typography, Button, Paper } from "@mui/material";

interface LayoutProps {
  onToggleColorMode: () => void;
}

export const Layout: FC<LayoutProps> = ({ onToggleColorMode }) => {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        bgcolor: theme.palette.background.default,
      }}
    >
      <AppBar position="static" color="primary">
        <Toolbar>
          <Typography
            variant="h6"
            component="div"
            sx={{ flexGrow: 1, color: theme.palette.primary.contrastText }}
          >
            ISXD File Viewer
          </Typography>
          <Button
            color="inherit"
            onClick={onToggleColorMode}
            sx={{ color: theme.palette.primary.contrastText }}
          >
            Toggle Theme
          </Button>
        </Toolbar>
      </AppBar>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          bgcolor: theme.palette.background.default,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 2,
            bgcolor: theme.palette.background.paper,
            borderRadius: 2,
          }}
        >
          <DirectoryNavigation />
        </Paper>
      </Box>
    </Box>
  );
};
