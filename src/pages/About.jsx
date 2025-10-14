import React from "react";
import { Typography, Box } from "@mui/material";

function About() {
  return (
    <Box sx={{ textAlign: "center" }}>
      <Typography variant="h4" gutterBottom>
        ℹ️ 프로젝트 소개
      </Typography>
      <Typography>커피 RAG + 재활용 이미지 분류 프로젝트</Typography>
    </Box>
  );
}

export default About;