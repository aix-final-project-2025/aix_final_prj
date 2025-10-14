import React from "react";
import { Drawer, List, ListItem, IconButton, Box } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import HomeIcon from "@mui/icons-material/Home";
import CoffeeIcon from "@mui/icons-material/Coffee";
import RecyclingIcon from "@mui/icons-material/Recycling";
import InfoIcon from "@mui/icons-material/Info";
import { useNavigate } from "react-router-dom";

function Sidebar({ open, setOpen, mobile }) {
  const navigate = useNavigate();

  const menuItems = [
    { icon: <HomeIcon sx={{ fontSize: 60, color: "#ff7043" }} />, path: "/" },         // 주황
    { icon: <CoffeeIcon sx={{ fontSize: 60, color: "#6d4c41" }} />, path: "/coffee" }, // 갈색
    { icon: <RecyclingIcon sx={{ fontSize: 60, color: "#388e3c" }} />, path: "/recycle" }, // 녹색
    { icon: <InfoIcon sx={{ fontSize: 60, color: "#1976d2" }} />, path: "/about" },    // 파랑
  ];

  return (
    <Drawer
      variant={mobile ? "temporary" : "persistent"}
      open={open}
      onClose={() => setOpen(false)}
      sx={{
        "& .MuiDrawer-paper": {
          width: 140, // 사이드바 폭 넓힘
          bgcolor: "#ffe0b2",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          position: "relative",
          padding: "20px 0",
        },
      }}
    >
      {/* 닫기 버튼 */}
      <Box sx={{ position: "absolute", top: 15, left: 15 }}>
        <IconButton onClick={() => setOpen(false)}>
          <CloseIcon sx={{ fontSize: 36 }} />
        </IconButton>
      </Box>

      {/* 메뉴 아이콘 리스트 */}
      <List
        sx={{
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 5, // 아이콘 간 간격 크게
        }}
      >
        {menuItems.map((item, index) => (
          <ListItem
            button
            key={index}
            onClick={() => {
              navigate(item.path);
              if (mobile) setOpen(false);
            }}
            sx={{
              justifyContent: "center",
              height: 100, // 아이콘 버튼 크기 키움
              borderRadius: "16px",
              "&:hover": {
                bgcolor: "#ffcc80",
              },
            }}
          >
            {item.icon}
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
}

export default Sidebar;