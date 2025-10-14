import React from "react";
import { Box, Typography } from "@mui/material";

function Home() {
  return (
    <Box sx={{ textAlign: "center" }}>
      <Typography variant="h3" gutterBottom>
        🎬 Welcome to Coffee & Recycle Project
      </Typography>
      <img
        src="https://placehold.co/800x400?text=Main+Visual"
        alt="main"
        style={{ maxWidth: "100%", borderRadius: "12px" }}
      />
    </Box>
  );
}

export default Home;