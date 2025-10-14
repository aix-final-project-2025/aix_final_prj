import React from "react";
import { Typography, Tabs, Tab, Box } from "@mui/material";

function Coffee() {
  const [tab, setTab] = React.useState(0);

  return (
    <Box>
      <Tabs value={tab} onChange={(e, v) => setTab(v)}>
        <Tab label="모델 1" />
        <Tab label="모델 2" />
        <Tab label="모델 3" />
        <Tab label="모델 4" />
        <Tab label="모델 5" />
      </Tabs>

      {tab === 0 && <Typography>☕ 모델 1 결과</Typography>}
      {tab === 1 && <Typography>☕ 모델 2 결과</Typography>}
      {tab === 2 && <Typography>☕ 모델 3 결과</Typography>}
      {tab === 3 && <Typography>☕ 모델 4 결과</Typography>}
      {tab === 4 && <Typography>☕ 모델 5 결과</Typography>}
    </Box>
  );
}

export default Coffee;