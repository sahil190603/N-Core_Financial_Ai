import { Card, Flex, message, Typography, Table, Row, Col } from "antd";
import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import MediaQueryHandler from "../components/Hooks/MediaQueryhandler";
import LoadingBar from "react-top-loading-bar";

const formatTitle = (text) =>
  text.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

const RenderContent = ({ content }) => {
  if (typeof content === "object" && content !== null) {
    if (Array.isArray(content)) {
      return (
        <ul>
          {content.map((item, idx) => (
            <li key={idx}>
              {typeof item === "object" && item !== null ? (
                <RenderContent content={item} />
              ) : (
                <span>{item}</span>
              )}
            </li>
          ))}
        </ul>
      );
    }

    return (
      <div>
        {Object.entries(content).map(([key, value]) => (
          <Card
            key={key}
            type="inner"
            title={formatTitle(key)}
            style={{ marginBottom: 8 }}
          >
            {Array.isArray(value) ? (
              <ul>
                {value.map((item, idx) => (
                  <li key={idx}>
                    {typeof item === "object" && item !== null ? (
                      <RenderContent content={item} />
                    ) : (
                      <span>{item}</span>
                    )}
                  </li>
                ))}
              </ul>
            ) : // Check if value is an object, if so, recursively call RenderContent

            typeof value === "object" && value !== null ? (
              <RenderContent content={value} />
            ) : (
              <span>{value}</span>
            )}
          </Card>
        ))}
      </div>
    );
  }

  return <span>{content}</span>;
};

const DynamicAnalysisDisplay = ({ selectedRecord }) => {
  if (!selectedRecord || !selectedRecord.AI_Response) {
    return null;
  }
  return (
    <Card
      style={{
        width: "100%",
        height: "calc(100vh - 153px)",
        overflowY: "auto",
      }}
    >
      <RenderContent content={selectedRecord.AI_Response} />
    </Card>
  );
};

const About = () => {
  const [loading, setLoading] = useState(false);
  const [financialData, setFinancialData] = useState([]);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const theme = localStorage.getItem("theme");
  const { isMobile } = MediaQueryHandler();

  const loadingBarRef = useRef(null);

  useEffect(() => {
    Fetch_chats();
  }, []);

  const truncateText = (text, wordLimit = 25) => {
    const words = text.split(" ");
    return words.length > wordLimit
      ? words.slice(0, wordLimit).join(" ") + "..."
      : text;
  };

  const columns = [
    {
      title: "User queries",
      dataIndex: "user_query",
      key: "user_query",
      render: (text) => (
        <div style={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
          {truncateText(text)}
        </div>
      ),
    },
  ];

  const tableData = financialData.map((item) => ({
    key: item.id,
    user_query: item.user_query,
    AI_Response: item.AI_Response,
  }));

  const Fetch_chats = async () => {
    setLoading(true);
    if (loadingBarRef.current) {
      loadingBarRef.current.continuousStart();
    }
    try {
      const response = await axios.get(
        "http://localhost:8000/api/Financial_ai_data/"
      );
      setTimeout(() => {
        setFinancialData(response.data);
        setLoading(false);
        if (loadingBarRef.current) {
          loadingBarRef.current.complete();
        }
      }, 2000);
    } catch {
      message.error("Unable to fetch data");
      setLoading(false);
      if (loadingBarRef.current) {
        loadingBarRef.current.complete();
      }
    }
  };

  const handleTableSelect = (selectedKeys, selectedRows) => {
    setSelectedRowKeys(selectedKeys);
    setSelectedRecord(selectedRows.length > 0 ? selectedRows[0] : null);
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "calc(100vh - 75px)",
      }}
    >
      {/* The top loading bar */}
      <LoadingBar
        ref={loadingBarRef}
        color={theme === "dark" ? "#13c2c2" : "#003eb3"}
        height={theme === "dark" ? "2px" : "3px"}
      />
      <Card
        style={{
          width: "100%",
          height: "100%",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
          overflowY: "auto",
          borderRadius: "5px",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            minHeight: "calc(100vh - 125px)",
            borderRadius: "5px",
            boxShadow:
              theme === "dark"
                ? "0 0 10px rgb(182, 182, 182)"
                : "0 4px 12px rgba(0, 0, 0, 0.54)",
          }}
        >
          <Row>
            <Col span={isMobile ? 24 : 10}>
              <div style={{ flex: "40%", padding: "10px"}}>
                <Table
                loading={loading}
                  dataSource={tableData}
                  columns={columns}
                  pagination={false}
                  scroll={{ y: "calc(100vh - 185px)" }}
                  bordered
                  size="small"
                  rowSelection={{
                    type: "radio",
                    selectedRowKeys,
                    onChange: handleTableSelect,
                  }}
    
                />
              </div>
            </Col>
            <Col
              span={isMobile ? 24 : 14}
              style={{ padding: "10px", marginTop: isMobile ? "10px" : "" }}
            >
              <Flex justify="center" align="center" style={{ height: "100%" }}>
                {selectedRecord ? (
                  <DynamicAnalysisDisplay selectedRecord={selectedRecord} />
                ) : (
                  <Typography.Title type="secondary" level={5}>
                    Select a record to view AI analysis
                  </Typography.Title>
                )}
              </Flex>
            </Col>
          </Row>
        </div>
      </Card>
    </div>
  );
};

export default About;
