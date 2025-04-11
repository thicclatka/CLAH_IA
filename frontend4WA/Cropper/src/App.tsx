import { createRoot } from "react-dom/client";
import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider, CssBaseline } from "@mui/material";
import { getTheme } from "./config/theme";
import { Layout } from "./config/layout";

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  const [mode, setMode] = useState<"light" | "dark">("dark");

  const toggleColorMode = () => {
    setMode((prevMode) => (prevMode === "light" ? "dark" : "light"));
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={getTheme(mode)}>
        <CssBaseline />
        <Layout onToggleColorMode={toggleColorMode} />
      </ThemeProvider>
    </QueryClientProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
