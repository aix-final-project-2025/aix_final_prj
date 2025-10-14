import React, { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { Box, IconButton } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import useMediaQuery from "@mui/material/useMediaQuery";

import Sidebar from "./components/Sidebar";
import Home from "./pages/Home";
import Coffee from "./pages/Coffee";
import Recycle from "./pages/Recycle";
import About from "./pages/About";

function App() {
  const [open, setOpen] = useState(false); // 첫 진입은 사이드바 닫힘
  const isMobile = useMediaQuery("(max-width:768px)");

  return (
    <Box sx={{ display: "flex", width: "100%", height: "100vh" }}>
      {/* 사이드바 */}
      <Sidebar open={open} setOpen={setOpen} mobile={isMobile} />

      {/* 좌측 상단 햄버거 버튼 (열기 전용) */}
      {!open && (
        <Box sx={{ position: "absolute", top: 10, left: 10 }}>
          <IconButton onClick={() => setOpen(true)} color="inherit">
            <MenuIcon />
          </IconButton>
        </Box>
      )}

      {/* 본문 */}
      <Box
        className="content"
        sx={{
          flexGrow: 1,
          p: 3,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/coffee" element={<Coffee />} />
          <Route path="/recycle" element={<Recycle />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Box>
    </Box>
  );
}

export default App;