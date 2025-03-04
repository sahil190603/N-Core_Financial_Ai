import {
  Card,
  Col,
  Input,
  Row,
  Checkbox,
  Button,
  Form,
  Spin,
  Divider,
  message,
  Select,
} from "antd";
import React, { useState } from "react";
import { Invest_option, Risk_level } from "../Constant";
import axios from "axios";

const Invest = () => {
  const [form] = Form.useForm();
  const theme = localStorage.getItem("theme");
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const onFinish = async (values) => {
    setLoading(true);
    setAnalysis(null);

    const selectedLabels = Invest_option.filter((option) =>
      values.investment_options.includes(option.key)
    ).map((option) => option.Value);

    const userMessage = `I have ${
      values.amount
    } and I want to invest in ${selectedLabels.join(
      ", "
    )}, I prefer Risk level ${values.risk_level}.`;
    try {
      const response = await axios.post(
        "http://localhost:8000/api/investment_pre/",
        {
          user_query: userMessage,
        }
      );
      setAnalysis(response.data.data.investment_strategy);
      message.success("Analysis completed successfully!");
    } catch (error) {
      message.error("API Error:", error);
    }
    setLoading(false);
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
            fontSize: "18px",
            fontWeight: "bold",
            display: "flex",
            marginBottom: "10px",
          }}
        >
          Investment Planner
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
              }}
            >
              <Form form={form} onFinish={onFinish} layout="vertical">
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                  }}
                >
                  <Form.Item
                    label="Investment Amount"
                    name="amount"
                    rules={[
                      {
                        required: true,
                        message: "Please enter an amount to invest!",
                      },
                      {
                        pattern: /^\d+$/,
                        message: "Only numeric values are allowed!",
                      },
                      {
                        validator: (_, value) =>
                          value && parseInt(value, 10) <= 50000
                            ? Promise.reject("Value must be greater than 50000")
                            : Promise.resolve(),
                      },
                    ]}
                  >
                    <Input
                      placeholder="Enter amount to invest"
                      style={{ width: 250 }}
                      maxLength={15}
                    />
                  </Form.Item>
                  <Form.Item
                    label="Investment Options"
                    name="investment_options"
                    rules={[
                      { required: true, message: "please select an option" },
                    ]}
                  >
                    <Checkbox.Group
                      options={Invest_option.map((option) => ({
                        label:
                          option.Value.charAt(0).toUpperCase() +
                          option.Value.slice(1),
                        value: option.key,
                      }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label="Risk Level"
                    name="risk_level"
                    rules={[
                      { required: true, message: "Please select a risk level" },
                    ]}
                    initialValue={"Low"}
                  >
                    <Select style={{ width: 150 }}>
                      {Risk_level.map((risk) => (
                        <Select.Option key={risk.key} value={risk.key}>
                          {risk.Value}
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
            marginTop: "20px",
            borderRadius: "5px",
            height: "calc(100vh - 323px)",
            boxShadow:
              theme === "dark"
                ? "0 0 10px rgb(182, 182, 182)"
                : "0 4px 12px rgba(0, 0, 0, 0.54)",
            overflow: "auto",
          }}
        >
          {loading ? (
            <Spin
              size="large"
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                width: "100%",
              }}
            />
          ) : analysis ? (
            <>
              <h3>Investment Strategy</h3>

              <Divider />

              <p>
                <strong>Budget:</strong> {analysis.budget}
              </p>

              <p>
                <strong>Risk Profile:</strong> {analysis.risk_profile}
              </p>

              <Divider />

              <h3>Recommended Allocations</h3>

              <ul>
                {Object.entries(analysis.recommended_allocations).map(
                  ([key, value]) => (
                    <li key={key}>
                      <strong>{key}:</strong> {value}
                    </li>
                  )
                )}
              </ul>

              <Divider />

              <h3>Justification</h3>

              <p>{analysis.justification}</p>
            </>
          ) : (
            <p>
              To see results, enter an amount and select investment options,
              then hit Analyze.
            </p>
          )}
        </Card>
      </Card>
    </div>
  );
};

export default Invest;
