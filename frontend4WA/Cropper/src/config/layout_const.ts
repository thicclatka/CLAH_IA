import { styled } from "@mui/material/styles";
import { ListItem, Box, Paper, List } from "@mui/material";

export const StyledListItem = styled(ListItem)(({ theme }) => ({
  padding: theme.spacing(0.5, 1),
  cursor: "pointer",
  "&:hover": {
    backgroundColor: theme.palette.action.hover,
    textDecoration: "underline",
  },
}));

export const DirectoryContainer = styled(Box)(({ theme }) => ({
  display: "flex",
  flexDirection: "row",
  gap: theme.spacing(2),
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
  height: "100vh",
}));

export const LeftColumn = styled(Box)(({ theme }) => ({
  width: "50%",
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(2),
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[1],
}));

export const RightColumn = styled(Box)(({ theme }) => ({
  width: "50%",
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(2),
}));

export const DirectoryHeader = styled(Box)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(1),
  padding: theme.spacing(1),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
}));

export const DirectoryInput = styled(Box)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(1),
  padding: theme.spacing(1),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
}));

export const Section = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[1],
  flex: 1,
  display: "flex",
  flexDirection: "column",
  gap: theme.spacing(0.5),
}));

export const DirectoryListContainer = styled(Box)(({ theme }) => ({
  maxHeight: "200px",
  overflowY: "auto",
  padding: 0,
  margin: 0,
  "& .MuiListItem-root": {
    minHeight: "40px",
  },
  "& .MuiListItemText-primary": {
    fontSize: "0.875rem",
  },
}));

export const ListContainer = styled(List)(({}) => ({
  maxHeight: "30px",
  padding: 0,
  margin: 0,
  "& .MuiListItemText-primary": {
    fontSize: "0.75rem", // caption size
  },
}));

export const FileViewerContainer = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(2),
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[1],
}));
