import { Card, Col, Row, Button, Divider, Spin, message, Upload } from "antd";
import React, { useState } from "react";
import axios from "axios";
import {
  InboxOutlined,
  LoadingOutlined,
  SendOutlined,
} from "@ant-design/icons";

const { Dragger } = Upload;

const DebtAnalysis = () => {
  // State now holds files instead of text strings
  const [cashFlowFile, setCashFlowFile] = useState(null);
  const [balanceSheetFile, setBalanceSheetFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const theme = localStorage.getItem("theme");

  const analyzeDebt = async () => {
    if (!cashFlowFile || !balanceSheetFile) {
      message.error("Both cash flow and balance sheet files are required!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append(
      "cash_flow",
      cashFlowFile.originFileObj ? cashFlowFile.originFileObj : cashFlowFile
    );
    formData.append(
      "balance_sheet",
      balanceSheetFile.originFileObj
        ? balanceSheetFile.originFileObj
        : balanceSheetFile
    );
    try {
      const response = await axios.post(
        "http://localhost:8000/api/pre_debt/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.status === 200) {
        setAnalysis(response.data.analysis);
      } else {
        message.error(response.data.error || "Error analyzing debt.");
      }
    } catch (error) {
      message.error("Network error. Please try again.");
    } finally {
      setTimeout(() => {
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
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "10px",
          }}
        >
          <div style={{ fontSize: "18px", fontWeight: "bold" }}>
            Debt Analyzer
          </div>
          <Button
            icon={loading ? <LoadingOutlined /> : <SendOutlined />}
            type="primary"
            onClick={analyzeDebt}
            disabled={!cashFlowFile || !balanceSheetFile || loading}
          >
            {loading ? "Analyzing.." : "Analyze"}
          </Button>
        </div>

        <Row gutter={16}>
          <Col span={12}>
            <Card
              style={{
                borderRadius: "5px",
                boxShadow:
                  theme === "dark"
                    ? "0 0 10px rgb(182, 182, 182)"
                    : "0 4px 12px rgba(0, 0, 0, 0.54)",
              }}
            >
              <Dragger
                beforeUpload={(file) => {
                  if (file.type !== "application/pdf") {
                    message.error("Only PDF files are allowed.");
                    return Upload.LIST_IGNORE;
                  }
                  setCashFlowFile(file);
                  return false; 
                }}
                onChange={(info) => {
                  if (info.fileList.length > 1) {
                    message.warning("Only one file can be uploaded at a time.");
                    return;
                  }
                  if (info.fileList.length === 1) {
                    setCashFlowFile(info.fileList[0]);
                  } else {
                    setCashFlowFile(null);
                  }
                }}
                onRemove={() => {
                  setCashFlowFile(null);
                }}
                accept="application/pdf"
                multiple={false}
                fileList={cashFlowFile ? [cashFlowFile] : []}
              >
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">
                  Click or drag file to upload Cash Flow Statement
                </p>
                <p className="ant-upload-hint">Only PDF file is allowed.</p>
              </Dragger>
            </Card>
          </Col>

          <Col span={12}>
            <Card
              style={{
                borderRadius: "5px",
                boxShadow:
                  theme === "dark"
                    ? "0 0 10px rgb(182, 182, 182)"
                    : "0 4px 12px rgba(0, 0, 0, 0.54)",
              }}
            >
              <Dragger
                beforeUpload={(file) => {
                  if (file.type !== "application/pdf") {
                    message.error("Only PDF files are allowed.");
                    return Upload.LIST_IGNORE;
                  }
                  setBalanceSheetFile(file);
                  return false;
                }}
                onChange={(info) => {
                  if (info.fileList.length > 1) {
                    message.warning("Only one file can be uploaded at a time.");
                    return;
                  }
                  if (info.fileList.length === 1) {
                    setBalanceSheetFile(info.fileList[0]);
                  } else {
                    setBalanceSheetFile(null);
                  }
                }}
                onRemove={() => {
                  setBalanceSheetFile(null);
                }}
                accept="application/pdf"
                multiple={false}
                fileList={balanceSheetFile ? [balanceSheetFile] : []}
              >
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">
                  Click or drag file to upload Balance Sheet
                </p>
                <p className="ant-upload-hint">Only PDF file is allowed.</p>
              </Dragger>
            </Card>
          </Col>
        </Row>
        <Row style={{ marginTop: "10px" }}>
          <Col span={24}>
            <Card
              style={{
                height: "calc(100vh - 416px)",
                overflowY: "auto",
                borderRadius: "5px",
                boxShadow:
                  theme === "dark"
                    ? "0 0 10px rgb(182, 182, 182)"
                    : "0 4px 12px rgba(0, 0, 0, 0.54)",
                display: "flex",
                flexDirection: "column",
                textAlign: "left",
              }}
            >
              {loading ? (
                <Spin
                size="large"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "calc(100vh - 500px)",
                    width: "100%",
                  }}
                />
              ) : analysis ? (
                <>
                  <h3>Debt Analysis</h3>
                  <Divider />
                  <p>
                    <strong>Debt-to-Equity Ratio:</strong>{" "}
                    {analysis.debt_analysis.leverage_ratios.debt_to_equity}
                  </p>
                  <p>
                    <strong>Interest Coverage Ratio:</strong>{" "}
                    {analysis.debt_analysis.leverage_ratios.interest_coverage}
                  </p>

                  <Divider />

                  <h3>Repayment Risks</h3>
                  <p>{analysis.debt_analysis.repayment_risks}</p>

                  <Divider />

                  <h3>Recommendations</h3>
                  <ul>
                    {analysis.debt_analysis.recommendations.map(
                      (rec, index) => (
                        <li key={index}>{rec}</li>
                      )
                    )}
                  </ul>

                  <Divider />

                  <h3>Cash Flow Improvement Strategies</h3>
                  <ul>
                    {analysis.debt_analysis.cash_flow_suggestions.map(
                      (suggestion, index) => (
                        <li key={index}>{suggestion}</li>
                      )
                    )}
                  </ul>
                </>
              ) : (
                <p>
                  To see results, upload Cash Flow and Balance Sheet files, then
                  hit Analyze.
                </p>
              )}
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default DebtAnalysis;
