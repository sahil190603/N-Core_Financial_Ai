import React, { useState } from "react";
import { Card, Select, Spin, Input, Button, Row, Col, message } from "antd";
import MediaQueryHandler from "../components/Hooks/MediaQueryhandler";
import { Tickers } from "../Constant";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import { CaretDownOutlined, CaretUpOutlined } from "@ant-design/icons";

const { Option } = Select;

function Stock() {
  const { isMobile } = MediaQueryHandler();
  const [selectedTicker, setSelectedTicker] = useState("RELIANCE.NS");
  const [days, setDays] = useState(30);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const theme = localStorage.getItem("theme");

  const fetchForecastData = async () => {
    setChartData(null);
    setMetrics(null);
    setLoading(true);
    try {
      const news = Tickers.find(
        (ticker) => ticker.Value === selectedTicker
      ).news;
      const Company = Tickers.find(
        (ticker) => ticker.Value === selectedTicker
      ).name;
      const response = await fetch(
        `http://localhost:8000/api/predict/?ticker=${selectedTicker}&days=${days}&news=${news}&Company=${Company}`
      );
      const data = await response.json();
      setChartData(data.forecast);

      setMetrics(data.metrics);

      if (data.forecast?.Close) {
        const closes = Object.values(data.forecast.Close);
        const changes = closes.slice(1).map((val, i) => val > closes[i]);
        const upPercent =
          (changes.filter(Boolean).length / changes.length) * 100;
        setMetrics((prev) => ({
          ...prev,
          upPercent: Math.round(upPercent),
          downPercent: Math.round(100 - upPercent),
        }));
      }
    } catch (error) {
      message.error(`Error fetching forecast data: ${error}`);
    }
    setLoading(false);
  };

  const prepareChartData = () => {
    if (!chartData) return null;

    const dates = Object.keys(chartData.Open);
    return {
      labels: dates,
      datasets: [
        {
          label: "Open",
          data: dates.map((date) => chartData.Open[date]),
          borderColor: "green",
          borderWidth: 2,
          fill: false,
        },
        {
          label: "Close",
          data: dates.map((date) => chartData.Close[date]),
          borderColor: "red",
          borderWidth: 2,
          fill: false,
        },
      ],
    };
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
          Stock price Predictor
        </div>
        <Row gutter={10}>
          <Col>
            <Select
              defaultValue={selectedTicker}
              style={{ width: 200, marginBottom: "20px" }}
              onChange={(value) => setSelectedTicker(value)}
            >
              {Tickers.map((ticker) => (
                <Option key={ticker.id} value={ticker.Value}>
                  {ticker.Value}
                </Option>
              ))}
            </Select>
          </Col>
          <Col>
            <Input
              type="number"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              style={{ width: 200, marginBottom: "20px" }}
              placeholder="Enter days"
            />
          </Col>
          <Col>
            <Button
              type="primary"
              onClick={fetchForecastData}
              style={{ marginBottom: "20px" }}
            >
              Submit
            </Button>
          </Col>
        </Row>
        {metrics && !loading ? (
          <>
            <Row
              gutter={16}
              style={{ display: "flex", flexWrap: "wrap", marginBottom: 20 }}
            >
              <Col span={isMobile ? 24 : 5} style={{ display: "flex" }}>
                <Card
                  title="News Sentiment"
                  style={{
                    width: "100%",
                    height: "100%",
                  }}
                >
                  <p>Sentiment: {metrics.news_sentiment?.sentiment}</p>
                  <div
                    style={{
                      color:
                        metrics.news_sentiment?.confidence >= 0.5
                          ? "green"
                          : "red",
                    }}
                  >
                    Confidence: {metrics.news_sentiment?.confidence}
                  </div>
                </Card>
              </Col>
              <Col span={isMobile ? 24 : 19} style={{ display: "flex" }}>
                <Card style={{ width: "100%", height: "100%" }}>
                  <p>Reason: {metrics.news_sentiment?.reason}</p>
                </Card>
              </Col>
            </Row>

            <Row
              gutter={16}
              style={{ display: "flex", flexWrap: "wrap", marginBottom: 20 }}
            >
              <Col span={isMobile ? 24 : 8} style={{ display: "flex" }}>
                <Card
                  title="Model Accuracy"
                  size="small"
                  style={{ width: "100%", height: "100%" }}
                >
                  <p>Accuracy: {metrics.accuracy}%</p>
                  <p>Precision: {metrics.precision}%</p>
                  <p>Recall: {metrics.recall}%</p>
                </Card>
              </Col>
              <Col span={isMobile ? 24 : 8} style={{ display: "flex" }}>
                <Card
                  title="Up and Down Probability"
                  size="small"
                  style={{ width: "100%", height: "100%" }}
                >
                  <p>
                    <CaretUpOutlined style={{ color: "green" }} /> :{" "}
                    {metrics.trend_probability?.up}%
                  </p>
                  <p>
                    <CaretDownOutlined style={{ color: "red" }} /> :{" "}
                    {metrics.trend_probability?.down}%
                  </p>
                </Card>
              </Col>
              <Col span={isMobile ? 24 : 8} style={{ display: "flex" }}>
                <Card
                  title="Forecast Trend"
                  size="small"
                  style={{ width: "100%", height: "100%" }}
                >
                  <p>
                    Up Days: {metrics.forecast_trend?.up_days}/
                    {metrics.forecast_trend?.total_days}
                  </p>
                  <p>
                    (
                    {(
                      (metrics.forecast_trend?.up_days /
                        metrics.forecast_trend?.total_days) *
                      100
                    ).toFixed(1)}
                    %)
                  </p>
                </Card>
              </Col>
            </Row>
          </>
        ) : null}

        <Card
          style={{
            borderRadius: "5px",
            height: "495px",
            boxShadow:
              theme === "dark"
                ? "0 0 10px rgb(182, 182, 182)"
                : "0 4px 12px rgba(0, 0, 0, 0.54)",
          }}
        >
          {!loading && !chartData && (
            <div
              style={{
                height: "430px",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                textAlign: "center",
              }}
            >
              Select Stock name and Enter desire days to predict Price.
            </div>
          )}
          {loading ? (
            <div
              style={{
                height: "400px",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                textAlign: "center",
              }}
            >
              <Spin size="large" />
            </div>
          ) : (
            chartData && (
              <div style={{ height: "400px" }}>
                <Line
                  data={prepareChartData()}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    devicePixelRatio: 2,
                    scales: {
                      y: {
                        title: {
                          display: true,
                          text: "Stock Price (₹)",
                          font: { size: 14 },
                        },
                        ticks: {
                          beginAtZero: false,
                          font: { size: 12 },
                        },
                      },
                      x: {
                        ticks: {
                          font: { size: 12 },
                        },
                      },
                    },
                  }}
                />
              </div>
            )
          )}
        </Card>
      </Card>
    </div>
  );
}

export default Stock;
