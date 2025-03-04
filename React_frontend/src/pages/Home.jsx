import React, { useState } from "react";
import { Avatar, Card, Col, Row, message, Spin, Upload, Button } from "antd";
import {
  SendOutlined,
  InboxOutlined,
  LoadingOutlined,
} from "@ant-design/icons";
import MediaQueryHandler from "../components/Hooks/MediaQueryhandler";

const formatTitle = (text) =>
  text.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

const RenderContent = ({ content }) => {
  if (typeof content === "object" && content !== null) {
    if (Array.isArray(content)) {
      return (
        <ul>
          {content.map((item, idx) => (
            <li key={idx}>
              <RenderContent content={item} />
            </li>
          ))}
        </ul>
      );
    }
    return (
      <div>
        {Object.entries(content).map(([key, value]) => (
          <div key={key} style={{ marginBottom: 8 }}>
            <strong>{formatTitle(key)}:</strong>
            <RenderContent content={value} />
          </div>
        ))}
      </div>
    );
  }
  return <span>{content}</span>;
};

const DynamicAnalysisDisplay = ({ analysisResponse }) => {
  if (!analysisResponse || typeof analysisResponse !== "object") {
    return null;
  }
  return (
    <>
      {Object.entries(analysisResponse).map(([sectionKey, sectionContent]) => (
        <Card
          key={sectionKey}
          type="inner"
          title={formatTitle(sectionKey)}
          style={{ marginBottom: 16 }}
        >
          <RenderContent content={sectionContent} />
        </Card>
      ))}
    </>
  );
};

function Home() {
  const [analysisResponse, setAnalysisResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const theme = localStorage.getItem("theme");
  const { isMobile } = MediaQueryHandler();

  const handleAnalyze = async () => {
    if (!file) {
      message.error("Please upload a PDF file first.");
      return;
    }
    setAnalysisResponse(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("pdf_file", file.originFileObj || file);
    try {
      const response = await fetch(
        "http://localhost:8000/api/process_financial_query/",
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) {
        throw new Error("Failed to process the PDF file");
      }
      const data = await response.json();
      setAnalysisResponse(data.data);
    } catch (error) {
      message.error("Error processing the file");
    } finally {
      setTimeout(async () => {
        setLoading(false);
      }, 3000);
    }
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
      <Card
        style={{
          textAlign: "center",
          borderRadius: "5px",
          width: "100%",
          height: "100%",
          flexDirection: "column",
          alignItems: "center",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
          overflowY: "auto",
        }}
      >
        <div
          style={{
            fontSize: "18px",
            fontWeight: "bold",
            display: "flex",
            marginBottom: "10px",
          }}
        >
          Balance Sheet And Cash Flow Analyzer
        </div>
        <Row gutter={16} style={{ height: "100%" }}>
          {/* Left Column: File Upload Section */}
          <Col span={isMobile ? 24 : 12} style={{ height: "100%" }}>
            <Card
              style={{
                height: "100%",
                borderRadius: "5px",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                minHeight: "calc(100vh - 165px)",
                boxShadow:
                  theme === "dark"
                    ? "0 0 10px rgb(182, 182, 182)"
                    : "0 4px 12px rgba(0, 0, 0, 0.54)",
              }}
            >
              <div>
                <Card.Meta
                  avatar={
                    <Avatar src="https://ui.shadcn.com/avatars/shadcn.jpg" />
                  }
                  title="Upload Financial Statement PDF"
                  description={
                    <div
                      style={{
                        maxHeight: "calc(100vh - 250px)",
                        overflowY: "auto",
                      }}
                    >
                      Upload your Balance Sheet or Cash Flow Statement in PDF
                      format for analysis.
                    </div>
                  }
                />
              </div>
              <div style={{ marginTop: "17%" }}>
                <Upload.Dragger
                  beforeUpload={(file) => {
                    if (file.type !== "application/pdf") {
                      message.error("Only PDF files are allowed.");
                      return Upload.LIST_IGNORE;
                    }
                    if (file) {
                      setFile(file);
                    }
                    return false;
                  }}
                  onChange={(info) => {
                    if (info.fileList.length > 1) {
                      info.fileList.splice(1);
                      message.warning(
                        "Only one file can be uploaded at a time."
                      );
                    }
                    if (info.fileList.length === 1) {
                      setFile(info.fileList[0]);
                    } else {
                      setFile(null);
                    }
                  }}
                  onRemove={() => setFile(null)}
                  accept="application/pdf"
                  multiple={false}
                  fileList={file ? [file] : []}
                >
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                  </p>
                  <p className="ant-upload-text">
                    Click or drag file to this area to upload
                  </p>
                  <p className="ant-upload-hint">
                    Support for a single upload. Only PDF files are allowed.
                  </p>
                </Upload.Dragger>
              </div>
              <div
                style={{
                  position: "absolute",
                  bottom: 5,
                  right: 0,
                  padding: "10px",
                }}
              >
                <Button
                  type="primary"
                  onClick={handleAnalyze}
                  disabled={!file}
                  icon={loading? <LoadingOutlined/> : <SendOutlined />}
                >
                  {loading ? "Analyzing.." : "Analyze" }
                </Button>
              </div>
            </Card>
          </Col>

          {/* Right Column: Generated Financial Analysis */}
          <Col
            span={isMobile ? 24 : 12}
            style={{ marginTop: isMobile ? 20 : 0 }}
          >
            <Card
              style={{
                height: "calc(100vh - 165px)",
                borderRadius: "5px",
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                boxShadow:
                  theme === "dark"
                    ? "0 0 10px rgb(182, 182, 182)"
                    : "0 4px 12px rgba(0, 0, 0, 0.54)",
              }}
            >
              <div>
                <Spin
                  spinning={loading}
                  size="large"
                  style={{
                    justifyContent: "center",
                    alignItems: "center",
                    height: "calc(100vh - 135px)",
                  }}
                >
                  {!loading && (
                    <>
                      {analysisResponse ? (
                        <DynamicAnalysisDisplay
                          analysisResponse={analysisResponse}
                        />
                      ) : (
                        <Card.Meta
                          title="Generated Financial Analysis"
                          description="No analysis generated yet."
                        />
                      )}
                    </>
                  )}
                </Spin>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
}

export default Home;
