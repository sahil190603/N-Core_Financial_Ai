import {
  Card,
  Col,
  Form,
  Row,
  Select,
  Button,
  Table,
  message,
  Progress,
} from "antd";
import React, { useState } from "react";
import axios from "axios";
import { Industries } from "../Constant";

const Market_S_C = () => {
  const [form] = Form.useForm();
  const theme = localStorage.getItem("theme");
  const [tableData, setTableData] = useState([]);
  const [totalMCAP, setTotalMCAP] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFormSubmit = async (values) => {
    try {
      setLoading(true);
      setTimeout(async () => {
        try {
          const response = await axios.get(
            "http://localhost:8000/api/industry-occupancy/",
            {
              params: { industry: values.industry_name },
            }
          );
          const resData = response.data;
          setTotalMCAP(resData.total_MCAP);
          setTableData(resData.data);
        } catch (error) {
          message.error(`API Error: ${error}`);
        } finally {
          setLoading(false);
        }
      }, 2000);
    } catch (error) {
      message.error(`Unexpected Error: ${error}`);
      setLoading(false);
    }
  };
  

  const columns = [
    { title: "Company Name", dataIndex: "compname", key: "compname" },
    { title: "Close Price", dataIndex: "CLOSE_PRICE", key: "CLOSE_PRICE" },
    { title: "Market Cap (Cr.)", dataIndex: "MCAP", key: "MCAP" },
    { title: "Symbol", dataIndex: "symbol", key: "symbol" },
    {
      title: "Market Share (%)",
      dataIndex: "occupancy",
      key: "occupancy",
      render: (occupancy) => (
        <Progress type="circle" percent={occupancy} size={40} />
      ),
    },
  ];

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
            display: "flex",
            justifyContent: "space-between",
            marginBottom: "10px",
          }}
        >
          <div
            style={{
              fontSize: "18px",
              fontWeight: "bold",
            }}
          >
            Market Share Analyzer
          </div>
          {totalMCAP && (
            <div style={{ fontSize: "18px", fontWeight: "bold" }}>
              Total Market Cap of : {totalMCAP} (Cr.)
            </div>
          )}
        </div>

        <Row>
          <Col span={24}>
            <Card
              style={{
                borderRadius: "5px",
                boxShadow:
                  theme === "dark"
                    ? "0 0 10px rgb(182, 182, 182)"
                    : "0 4px 12px rgba(0, 0, 0, 0.54)",
                marginBottom: "10px",
              }}
            >
              <Form form={form} layout="vertical" onFinish={handleFormSubmit}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <Form.Item
                    label="Industry"
                    name="industry_name"
                    rules={[
                      { required: true, message: "Please select an industry" },
                    ]}
                  >
                    <Select
                      placeholder="Select Industry"
                      style={{ width: 250 }}
                      showSearch
                      optionFilterProp="children"
                      filterOption={(input, option) =>
                        option.children
                          .toLowerCase()
                          .includes(input.toLowerCase())
                      }
                    >
                      {Industries.map((industry) => (
                        <Select.Option key={industry.key} value={industry.key}>
                          {industry.Value}
                        </Select.Option>
                      ))}
                    </Select>
                  </Form.Item>
                  <Form.Item style={{ textAlign: "center" }}>
                    <Button type="primary" htmlType="submit">
                      Analyze
                    </Button>
                  </Form.Item>
                </div>
              </Form>
            </Card>
          </Col>
        </Row>

        <Card
          style={{
            marginTop: "10px",
            borderRadius: "5px",
            height: "calc(100vh - 321px)",
            boxShadow:
              theme === "dark"
                ? "0 0 10px rgb(182, 182, 182)"
                : "0 4px 12px rgba(0, 0, 0, 0.54)",
            overflow: "auto",
          }}
        >
          <Table
            columns={columns}
            dataSource={tableData}
            loading={loading}
            rowKey="fincode"
            pagination={{ pageSize: 10 }}
          />
        </Card>
      </Card>
    </div>
  );
};

export default Market_S_C;
