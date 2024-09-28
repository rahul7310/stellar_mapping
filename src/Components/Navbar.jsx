import {
  AppBar,
  Box,
  Button,
  Divider,
  Drawer,
  IconButton,
  ListItemButton,
  ListItemText,
  Toolbar,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import CloseIcon from "@mui/icons-material/Close";
import { useState } from "react";
export default function Navbar({ setSelectedTab, selectedTab }) {
  const [open, setOpen] = useState(false);
  const toggleDrawer = (open) => (event) => {
    if (
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }
    setOpen(open);
  };

  return (
    <AppBar
      style={{ backgroundColor: "black", height: "10px", zIndex: "1" }}
      position="static"
    >
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          aria-label="open drawer"
          sx={{ mr: 2, display: { xs: "block", sm: "flex", md: "none" } }}
          onClick={toggleDrawer(true)}
        >
          <MenuIcon />
        </IconButton>
        <div
          style={{
            display: "flex",
            justifyContent: "space-evenly",
            width: "100%",
          }}
        >
          <Button
            onClick={() => setSelectedTab("Introduction")}
            color="inherit"
            sx={{
              display: { xs: "none", md: "block" },
              border:
                selectedTab === "Introduction" ? "solid 1px yellow" : "none",
            }}
          >
            Introduction
          </Button>
          <Button
            onClick={() => setSelectedTab("Data Exploration")}
            color="inherit"
            sx={{
              display: { xs: "none", md: "block" },
              border:
                selectedTab === "Data Exploration"
                  ? "solid 1px yellow"
                  : "none",
            }}
          >
            Data Exploration
          </Button>
          <Button
            onClick={() => setSelectedTab("Model Implemented")}
            color="inherit"
            sx={{
              display: { xs: "none", md: "block" },
              border:
                selectedTab === "Model Implemented"
                  ? "solid 1px yellow"
                  : "none",
            }}
          >
            Model Implemented
          </Button>
          <Button
            onClick={() => setSelectedTab("Conclusion")}
            color="inherit"
            sx={{
              display: { xs: "none", md: "block" },
              border:
                selectedTab === "Conclusion" ? "solid 1px yellow" : "none",
            }}
          >
            Conclusion
          </Button>
          <Button
            onClick={() => setSelectedTab("Team")}
            color="inherit"
            sx={{
              display: { xs: "none", md: "block" },
              border: selectedTab === "Team" ? "solid 1px yellow" : "none",
            }}
          >
            Team
          </Button>
        </div>

        <Drawer anchor="left" open={open} onClose={toggleDrawer(false)}>
          <Box
            sx={{
              p: 2,
              height: 1,
              backgroundColor: "black",
              color: "ivory",
            }}
          >
            <IconButton onClick={toggleDrawer(false)} sx={{ mb: 2 }}>
              <CloseIcon />
            </IconButton>
            <Divider sx={{ mb: 2 }} />
            <Box sx={{ mb: 2 }}>
              <ListItemButton>
                <ListItemText
                  onClick={() => {
                    setOpen(false);
                    setSelectedTab("Introduction");
                  }}
                  style={{ color: selectedTab === "Introduction" && "yellow" }}
                  primary="Introduction"
                />
              </ListItemButton>
              <ListItemButton
                onClick={() => {
                  setOpen(false);
                  setSelectedTab("Data Exploration");
                }}
              >
                <ListItemText
                  style={{
                    color: selectedTab === "Data Exploration" && "yellow",
                  }}
                  primary="Data Exploration"
                />
              </ListItemButton>
              <ListItemButton
                onClick={() => {
                  setOpen(false);
                  setSelectedTab("Model Implemented");
                }}
              >
                <ListItemText
                  style={{
                    color: selectedTab === "Model Implemented" && "yellow",
                  }}
                  primary="Model Implemented"
                />
              </ListItemButton>
              <ListItemButton
                onClick={() => {
                  setOpen(false);
                  setSelectedTab("Conclusion");
                }}
              >
                <ListItemText
                  style={{ color: selectedTab === "Conclusion" && "yellow" }}
                  primary="Conclusion"
                />
              </ListItemButton>
              <ListItemButton
                onClick={() => {
                  setOpen(false);
                  setSelectedTab("Team");
                }}
              >
                <ListItemText
                  style={{ color: selectedTab === "Team" && "yellow" }}
                  primary="Team"
                />
              </ListItemButton>
            </Box>
          </Box>
        </Drawer>
      </Toolbar>
    </AppBar>
  );
}
