import React from "react";
import { Routes, Route } from "react-router-dom"; 
import Home from "../pages/Home";
import About from "../pages/About";
import Stock from "../pages/Stock";
import Debt from "../pages/Debt";
import Invest from "../pages/Invest";
import Market_S_C from "../pages/Market_S_C";

function MainRoute () {
  return (
    <div >
      <Routes  >
        <Route path="/" element={<Home />} />
        <Route path="/Contact_us" element={<About/>} />
        <Route path="/Predict" element={<Stock />} />
        <Route path="/Debt-pre" element={<Debt />}/>
        <Route path="/Invest" element={<Invest />} />
        <Route path="/msc" element={<Market_S_C />}/>
      </Routes>
    </div>
  );
};

export default MainRoute;
