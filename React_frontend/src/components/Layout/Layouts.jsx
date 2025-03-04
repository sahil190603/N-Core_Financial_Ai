import React, { useState, useEffect } from "react";
import {
  Menu,
  Layout,
  ConfigProvider,
  Drawer,
  Button,
  Divider,
} from "antd";
import { theme } from "antd";
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MenuOutlined,
  HomeOutlined,
  StockOutlined,
  UnorderedListOutlined,
  FundOutlined,
  InfoCircleOutlined,
  PropertySafetyOutlined,
} from "@ant-design/icons";
import ModeToggle from "../Generic/ModeToggle";
import MainRoute from "../../Route/route";
import MediaQueryHandler from "../Hooks/MediaQueryhandler";
import { useNavigate, useLocation } from "react-router-dom";
import useThemeDetector from "../Hooks/ThemeDetector";

const { Header, Content, Sider } = Layout;

function Layouts() {
  const isDarkTheme = useThemeDetector();
  const { defaultAlgorithm, darkAlgorithm } = theme;

  const storedTheme = localStorage.getItem("theme") || isDarkTheme;
  const [isDarkMode, setIsDarkMode] = useState(storedTheme === "dark");
  const [collapsed, setCollapsed] = useState(false);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [selectedKey, setSelectedKey] = useState();

  const { isMobile } = MediaQueryHandler();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
      setIsDarkMode(savedTheme === "dark");
    }
  }, []);

  useEffect(() => {
    if (location.pathname === "/contact_us") {
      setSelectedKey("6");
    } else if (location.pathname === "/msc") {
      setSelectedKey("5");
    } else if (location.pathname === "/Predict") {
      setSelectedKey("3");
    } else if (location.pathname === "/Debt-pre") {
      setSelectedKey("4");
    } else if (location.pathname === "/Invest") {
      setSelectedKey("2");
    } else {
      setSelectedKey("1");
    }
  }, [location.pathname]);

  const handleToggle = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem("theme", newTheme ? "dark" : "light");
  };

  const handleCollapse = () => {
    setCollapsed(!collapsed);
  };

  const toggleDrawer = () => {
    setDrawerVisible(!drawerVisible);
  };

  const handleMenuClick = (key) => {
    setSelectedKey(key);
    if (key === "1") {
      navigate("/");
    } else if (key === "2") {
      navigate("/Invest");
    } else if (key === "3") {
      navigate("/Predict");
    } else if (key === "4") {
      navigate("/Debt-pre");
    } else if (key === "5") {
      navigate("/msc");
    } else if (key === "6") {
      navigate("/contact_us");
    }
    if (isMobile) {
      toggleDrawer();
    }
  };
  const getMenuItems = () => {
    return [
      {
        key: "1",
        icon: <HomeOutlined />,
        label: "BS & CF Analysis",
        title: "Balance sheet / Cash Flow Analysis", // Add title prop here
        onClick: () => handleMenuClick("1"),
      },
      {
        key: "2",
        icon: <InfoCircleOutlined />,
        label: "Investment Planner",
        title: "Investment Planner",
        onClick: () => handleMenuClick("2"),
      },
      {
        key: "3",
        icon: <StockOutlined />,
        label: "Stock Prediction",
        title: "Stock Prediction",
        onClick: () => handleMenuClick("3"),
      },
      {
        key: "4",
        icon: <FundOutlined />,
        label: "Debt Analysis",
        title: "Debt Analysis",
        onClick: () => handleMenuClick("4"),
      },
      {
        key: "5",
        icon: <PropertySafetyOutlined />,
        label: "Market Share",
        title: "Market Share",
        onClick: () => handleMenuClick("5"),
      },
      {
        key: "6",
        icon: <UnorderedListOutlined />,
        label: "History",
        title: "History",
        onClick: () => handleMenuClick("6"),
      },
    ];
  };
  return (
    <ConfigProvider
      theme={{
        algorithm: isDarkMode ? darkAlgorithm : defaultAlgorithm,
      }}
    >
      <Layout style={{ minHeight: "100vh" }}>
        <Header
          style={{
            position: "fixed",
            width: "100%",
            zIndex: 1,
            display: "flex",
            alignItems: "center",
            backgroundColor: isDarkMode ? "#002140" : "#4096ff",
            padding: "0 22px",
            height: "55px",
          }}
        >
          <div className="logo">N-Core_Financial_Ai</div>

          <div
            style={{
              marginLeft: "auto",
              display: "flex",
              alignItems: "center",
            }}
          >
            <ModeToggle isDarkMode={isDarkMode} onToggle={handleToggle} />
            {isMobile && (
              <Button
                type="text"
                icon={
                  <MenuOutlined style={{ color: "white", fontSize: "18px" }} />
                }
                onClick={toggleDrawer}
                style={{ marginLeft: "10px" }}
              />
            )}
          </div>
        </Header>

        <Layout style={{ marginTop: 55 }}>
          {/* Sidebar for desktop */}
          {!isMobile && (
            <Sider
              collapsible
              width={180}
              collapsed={collapsed}
              onCollapse={handleCollapse}
              collapsedWidth={55}
              trigger={
                collapsed ? (
                  <Button
                    variant={"outlined"}
                    style={{ backgroundColor: "#002140" }}
                  >
                    <MenuUnfoldOutlined
                      style={{ color: "white", fontSize: "18px" }}
                    />
                  </Button>
                ) : (
                  <Button
                    variant={"outlined"}
                    style={{ backgroundColor: "#002140" }}
                  >
                    <MenuFoldOutlined
                      style={{ color: "white", fontSize: "18px" }}
                    />
                  </Button>
                )
              }
              style={{
                height: "100vh",
                position: "fixed",
                left: 0,
                top: 55,
                backgroundColor: isDarkMode ? "#001529" : "#ffffff",
              }}
            >
              <Menu
                selectedKeys={[selectedKey]}
                mode="inline"
                style={{ height: "100%" }}
                items={getMenuItems()}
              ></Menu>

              <Divider type="horizontal" style={{ margin: "10px 0" }} />
            </Sider>
          )}

          {/* Drawer for mobile */}
          {isMobile && (
            <Drawer
              title="Menu"
              placement="left"
              onClose={toggleDrawer}
              open={drawerVisible}
              styles={{
                body: { padding: 0 },
                header: { backgroundColor: isDarkMode ? "#001529" : "#ffffff" },
              }}
              width={200}
            >
              <Menu
                selectedKeys={[selectedKey]}
                mode="inline"
                style={{ height: "100%" }}
                items={getMenuItems()}
              ></Menu>
            </Drawer>
          )}

          <Layout
            style={{
              marginLeft: isMobile ? 0 : collapsed ? 55 : 180,
              padding: "10px",
              transition: "0.2s",
            }}
          >
            <Content
              style={{
                height: "100%",
                width: "100%",
              }}
            >
              <MainRoute />
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}

export default Layouts;
