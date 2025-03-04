import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import fitz
import requests
import numpy as np
import yfinance as yf
from openai import OpenAI
from keras import Sequential
from bs4 import BeautifulSoup
from rest_framework import viewsets
from django.http import JsonResponse
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dense, Input
from .models import FinancialAI, MarketOccupancy
from sklearn.linear_model import LogisticRegression
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from .serializers import FinancialAISerializer, MarketOccupancySerializers
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile



class FinancialAIViewset(viewsets.ModelViewSet):
    queryset = FinancialAI.objects.all().order_by("-id")
    serializer_class = FinancialAISerializer

class MarketOccupancySerializers(viewsets.ModelViewSet):
    queryset = MarketOccupancy.objects.all()
    serializer_class = MarketOccupancySerializers

@csrf_exempt
def process_financial_query(request):
    if request.method == "POST":
        user_query = None

        # Check if a PDF file was uploaded
        if "pdf_file" in request.FILES:
            pdf_file = request.FILES["pdf_file"]
            # Save the file temporarily
            temp_path = default_storage.save(f"temp/{pdf_file.name}", ContentFile(pdf_file.read()))
            try:
                # Use a context manager to ensure the PDF is closed after processing
                with fitz.open(default_storage.path(temp_path)) as doc:
                    extracted_text = "\n".join(page.get_text("text") for page in doc)
                # After the document is closed, delete the temporary file
                default_storage.delete(temp_path)
            except Exception as e:
                return JsonResponse({"error": f"Failed to extract text from PDF: {str(e)}"}, status=500)
            user_query = extracted_text
        else:
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON payload."}, status=400)
            user_query = data.get("user_query")
            if not user_query:
                return JsonResponse({"error": "The 'user_query' field is required."}, status=400)

        # Configure OpenAI client for Azure AI model
        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key="Replace with Your github Pat token."
        )

        system_prompt = """You are a highly intelligent and detail-oriented Personal Financial Assistant specializing in financial analysis. Your role is to analyze financial reports, assess financial health, and provide actionable insights while maintaining structured and JSON-formatted responses.

        ### Capabilities:
        1. **Financial Report Analysis:**
           - Analyze balance sheets, income statements, and cash flow statements.
           - Highlight key financial metrics (profitability, liquidity, solvency, efficiency).
           - Identify trends, anomalies, and potential financial risks.
        
        2. **Descriptive & Insightful Explanations:**
           - Provide plain-language explanations of financial data.
           - Avoid technical jargon.
           - Offer actionable recommendations.
        
        3. **Advanced Financial Insights:**
           - **Balance Sheet Analysis:** Assess assets, liabilities, and equity structure.
           - **Cash Flow Analysis:** Evaluate cash movements in operating, investing, and financing activities.
           - **Debt Analysis:** Examine debt structure, leverage ratios, and repayment risks.
           - **Stock Price Prediction (Trend-Based):** Analyze historical trends, volatility, and financial indicators.
           - **Market Share Analysis (Country-wise):** Assess company market position based on available financial data.
        
        ### Scope of Responses:
        - **Balance Sheet Analysis** (Assets, Liabilities, Equity)
        - **Income Statement Analysis** (Revenue, Expenses, Profitability)
        - **Cash Flow Analysis** (Operating, Investing, Financing Activities)
        - **Debt & Leverage Analysis** (Debt-to-Equity, Interest Coverage)
        - **Financial Ratios** (ROE, ROA, Current Ratio, P/E)
        - **Stock Trend Analysis** (Based on historical performance and fundamental factors)
        - **Market Share Insights** (Competitive position country-wise)
        
        ### Limitations:
        - **No Specific Investment Advice** (e.g., "Buy/Sell this stock").
        - **No Real-Time or Live Market Data** (All insights are based on provided or historical data).
        - **No Non-Financial Topics.**
        
        ### Response Structures:
        
        #### **Financial Analysis Response:**
        ```json
        {
          "analysis_summary": "<summary>",
          "key_findings": ["<finding1>", "<finding2>"],
          "recommendations": ["<recommendation1>", "<recommendation2>"]
        }
        Stock Price Prediction Response:
        json
        {
          "stock_trend_analysis": {
            "historical_trend": "<summary>",
            "key_factors": ["<factor1>", "<factor2>"],
            "risk_assessment": "<risk level>",
            "projection_summary": "<potential trend>"
          }
        }
        Market Share Response:
        json
        {
          "company_market_share": {
            "global_position": "<summary>",
            "country_wise_breakdown": {
              "USA": "<percentage>",
              "Europe": "<percentage>",
              "Asia": "<percentage>"
            },
            "competitive_risks": ["<risk1>", "<risk2>"]
          }
        }
        Out-of-Scope Response:
        json
        {
          "error": "Out of scope",
          "message": "I'm a Personal Financial Assistant! That request is beyond my expertise."
        }
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": extracted_text}
                ],
                temperature=1,
                max_tokens=4096,
                top_p=1
            )

            answer_text = response.choices[0].message.content

            # Clean the response to extract JSON
            if answer_text.startswith("```json") and answer_text.endswith("```"):
                # Remove ```json and trailing ```
                answer_text = answer_text[7:-3].strip()

            # Attempt to parse JSON response
            try:
                answer_json = json.loads(answer_text)
            except json.JSONDecodeError:
                answer_json = {"error": "Invalid JSON response",
                               "raw_response": answer_text}

            # Save to database
            FinancialAI.objects.create(
                user_query=extracted_text,
                AI_Response=answer_json
            )

            return JsonResponse({"data": answer_json}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed."}, status=405)


def analyze_news(news_text):
    """
    Analyze news sentiment using Azure Gemini API.
    Returns a dictionary with keys: sentiment, confidence, reason.
    """
    try:
        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(
                "Replace with Your github Pat token.")
        )
    except Exception as e:
        return {"error": "Error initializing client: " + str(e)}

    system_msg = SystemMessage(
        """You are a financial and business news analysis chatbot. Your task is to analyze financial or business-related news and classify its sentiment as Positive, Negative, or Intermediate based on its potential impact.
        
Your response must be in valid JSON format with the following structure:
{
  "sentiment": "Positive" | "Negative" | "Intermediate",
  "confidence": 0.0 - 1.0,
  "reason": "Brief explanation of why the sentiment was assigned"
}
Positive: If the news indicates economic growth, profit increase, market stability, or beneficial policies.
Negative: If the news suggests financial losses, market downturns, company bankruptcies, or adverse regulations.
Intermediate: If the impact is mixed, uncertain, or neutral."""
    )
    user_msg = UserMessage(news_text)
    try:
        sentiment_response = client.complete(
            messages=[system_msg, user_msg],
            model="gpt-4o",
            temperature=1,
            max_tokens=4096,
            top_p=1
        )
        sentiment_json_str = sentiment_response.choices[0].message.content.strip(
        )
        # Clean formatting artifacts
        sentiment_json_str = sentiment_json_str.replace(
            '```json', '').replace('```', '').strip()
        if sentiment_json_str.startswith('"') and sentiment_json_str.endswith('"'):
            sentiment_json_str = sentiment_json_str[1:-1]
        return json.loads(sentiment_json_str)
    except Exception as e:
        return {"error": "News analysis failed: " + str(e)}


@csrf_exempt
def predict_stock(request):
    if request.method == 'GET':
        # Read parameters
        ticker_value = request.GET.get('ticker', 'RELIANCE.NS')
        forecast_days = request.GET.get('days', 7)
        news_source = request.GET.get('news', 'RI')
        Company = request.GET.get('Company', 'reliance-industries-share-price')

        try:
            forecast_days = int(forecast_days)
            if forecast_days <= 0:
                return JsonResponse({'error': 'Days parameter must be greater than zero'})
        except ValueError:
            return JsonResponse({'error': 'Invalid days parameter'})

        # Fetch stock data
        try:
            df = yf.download(tickers=ticker_value, period='2y',
                             interval='1d', auto_adjust=False)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except Exception as e:
            return JsonResponse({'error': str(e)})

        # ------------------ Logistic Regression Classifier ------------------
        df_class = df.copy()
        df_class['Price_Change'] = df_class['Close'].diff()
        df_class['Actual_Movement'] = np.where(
            df_class['Price_Change'] > 0, 1, 0)
        df_class.dropna(inplace=True)
        features = ['Open', 'High', 'Low', 'Volume']
        X_cls = df_class[features]
        y_cls = df_class['Actual_Movement']
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_cls, y_cls, test_size=0.3, random_state=42
        )
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_cls, y_train_cls)
        lr_preds = lr_model.predict(X_test_cls)
        # Classification metrics
        cm = confusion_matrix(y_test_cls, lr_preds)
        accuracy = accuracy_score(y_test_cls, lr_preds)
        precision = precision_score(y_test_cls, lr_preds)
        recall = recall_score(y_test_cls, lr_preds)
        # ------------------ End Logistic Regression ------------------

        # ------------------ LSTM Forecasting ------------------
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Open', 'High', 'Close']])
        look_back = 60
        X, y_open, y_high, y_close = [], [], [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i+look_back])
            y_open.append(scaled_data[i+look_back, 0])
            y_high.append(scaled_data[i+look_back, 1])
            y_close.append(scaled_data[i+look_back, 2])
        X = np.array(X)
        y_open = np.array(y_open)
        y_high = np.array(y_high)
        y_close = np.array(y_close)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_open_train, y_open_test = y_open[:split], y_open[split:]
        y_high_train, y_high_test = y_high[:split], y_high[split:]
        y_close_train, y_close_test = y_close[:split], y_close[split:]

        def create_model():
            model = Sequential([
                Input(shape=(look_back, 3)),
                LSTM(units=100, return_sequences=True),
                LSTM(units=100),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model_open = create_model()
        model_high = create_model()
        model_close = create_model()

        model_open.fit(X_train, y_open_train, epochs=20, batch_size=32,
                       validation_data=(X_test, y_open_test), verbose=0)
        model_high.fit(X_train, y_high_train, epochs=20, batch_size=32,
                       validation_data=(X_test, y_high_test), verbose=0)
        model_close.fit(X_train, y_close_train, epochs=20, batch_size=32,
                        validation_data=(X_test, y_close_test), verbose=0)

        last_sequence = scaled_data[-look_back:]
        forecast_open, forecast_high, forecast_close = [], [], []
        for _ in range(forecast_days):
            pred_open = model_open.predict(
                last_sequence.reshape(1, look_back, 3), verbose=0)
            pred_high = model_high.predict(
                last_sequence.reshape(1, look_back, 3), verbose=0)
            pred_close = model_close.predict(
                last_sequence.reshape(1, look_back, 3), verbose=0)

            forecast_open.append(scaler.inverse_transform(
                [[pred_open[0][0], 0, 0]])[0][0])
            forecast_high.append(scaler.inverse_transform(
                [[0, pred_high[0][0], 0]])[0][1])
            forecast_close.append(scaler.inverse_transform(
                [[0, 0, pred_close[0][0]]])[0][2])

            new_row = np.array(
                [pred_open[0][0], pred_high[0][0], pred_close[0][0]])
            last_sequence = np.vstack((last_sequence[1:], new_row))
        # ------------------ End LSTM Forecasting ------------------

        # ------------------ Trend Prediction ------------------
        latest_data = df.iloc[-1][features].values.reshape(1, -1)
        trend_prob = lr_model.predict_proba(latest_data)[0]
        forecast_dates = [(datetime.today() + timedelta(days=i)
                           ).strftime('%Y-%m-%d') for i in range(forecast_days)]
        # ------------------ End Trend Prediction ------------------

        # ------------------ News Sentiment Analysis ------------------
        news_sentiment_result = {}
        news_impact_percentage = 0  # Default impact
        if news_source != 'RI' and news_source:
            news_url = f"https://www.cnbctv18.com/market/stocks/{Company}/{news_source}/"
            try:
                response = requests.get(news_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                news_elements = soup.find_all(class_='news_sec')
                news_text_combined = "\n".join(
                    ne.text.strip() for ne in news_elements if ne.text.strip())
                if news_text_combined:
                    news_sentiment_result = analyze_news(news_text_combined)
                    sentiment = news_sentiment_result.get(
                        'sentiment', 'Intermediate')
                    if sentiment == 'Positive':
                        news_impact_percentage = 0.03
                    elif sentiment == 'Negative':
                        news_impact_percentage = -0.02
                    elif sentiment == 'Intermediate':
                        news_impact_percentage = 0.01
                else:
                    news_sentiment_result = {"error": "No news content"}
            except requests.exceptions.RequestException as e:
                news_sentiment_result = {
                    "error": "RequestException", "details": str(e)}
            except Exception as e:
                news_sentiment_result = {
                    "error": "ProcessingError", "details": str(e)}
        else:
            news_sentiment_result = {"info": "News analysis skipped"}
        # ------------------ End News Sentiment Analysis ------------------

        # Adjust forecast close based on news impact
        if news_impact_percentage != 0:
            forecast_close_adjusted = {
                date: price * (1 + news_impact_percentage) for date, price in zip(forecast_dates, forecast_close)}
        else:
            forecast_close_adjusted = dict(zip(forecast_dates, forecast_close))

        # ------------------ Price Movement Metrics ------------------
        # Use original High data for price movement trends
        last_actual_high = float(df['High'].iloc[-1].item())
        forecast_movement = [1 if float(
            fc) > last_actual_high else 0 for fc in forecast_high]
        percentage_up = (sum(forecast_movement) / len(forecast_movement)) * 100
        percentage_down = 100 - percentage_up
        up_changes = [((float(fc) - last_actual_high) / last_actual_high * 100)
                      for fc in forecast_high if float(fc) > last_actual_high]
        down_changes = [((float(fc) - last_actual_high) / last_actual_high * 100)
                        for fc in forecast_high if float(fc) <= last_actual_high]
        avg_change_up = np.mean(up_changes) if up_changes else 0
        avg_change_down = np.mean(down_changes) if down_changes else 0
        # ------------------ End Price Movement Metrics ------------------

        response_data = {
            'ticker': ticker_value,
            'forecast': {
                'Open': dict(zip(forecast_dates, forecast_open)),
                'High': dict(zip(forecast_dates, forecast_high)),
                'Close': forecast_close_adjusted
            },
            'metrics': {
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'confusion_matrix': cm.tolist(),
                'trend_probability': {
                    'up': round(trend_prob[1] * 100, 2),
                    'down': round(trend_prob[0] * 100, 2)
                },
                'forecast_trend': {
                    'up_days': sum(1 for v in forecast_close if v > df['Close'].iloc[-1].item()),
                    'total_days': forecast_days
                },
                'news_sentiment': news_sentiment_result,
                'price_movement': {
                    'percentage_up': round(percentage_up, 2),
                    'percentage_down': round(percentage_down, 2),
                    'average_change_up': round(avg_change_up, 2),
                    'average_change_down': round(avg_change_down, 2)
                }
            }
        }
        return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def analyze_debt(request):
    if request.method == "POST":
        # Check that both files are provided
        if "balance_sheet" not in request.FILES or "cash_flow" not in request.FILES:
            return JsonResponse({"error": "Both 'balance_sheet' and 'cash_flow' files are required."}, status=400)
        
        balance_sheet_file = request.FILES["balance_sheet"]
        cash_flow_file = request.FILES["cash_flow"]

        bs_temp_path = default_storage.save(f"temp/{balance_sheet_file.name}", ContentFile(balance_sheet_file.read()))
        cf_temp_path = default_storage.save(f"temp/{cash_flow_file.name}", ContentFile(cash_flow_file.read()))
        
        try:
            # Use context managers to ensure PDFs are closed after extraction
            with fitz.open(default_storage.path(bs_temp_path)) as doc_bs:
                balance_sheet_text = "\n".join(page.get_text("text") for page in doc_bs)
            with fitz.open(default_storage.path(cf_temp_path)) as doc_cf:
                cash_flow_text = "\n".join(page.get_text("text") for page in doc_cf)
        except Exception as e:
            return JsonResponse({"error": f"Failed to extract text from PDF: {str(e)}"}, status=500)
        finally:
            # Delete temporary files
            try:
                default_storage.delete(bs_temp_path)
            except Exception:
                pass
            try:
                default_storage.delete(cf_temp_path)
            except Exception:
                pass
        
        # Validate that extracted texts are not empty
        if not balance_sheet_text.strip() or not cash_flow_text.strip():
            return JsonResponse({"error": "Extracted text from one of the files is empty."}, status=400)
        
        # Combine financial data into a single input
        financial_data = f"Balance Sheet:\n{balance_sheet_text}\n\nCash Flow Statement:\n{cash_flow_text}"


        # System Prompt for AI Model
        system_prompt = SystemMessage(
            """You are a financial analyst. Analyze the provided cash flow statement and balance sheet to perform a comprehensive debt analysis. Evaluate the company’s leverage ratios, cash flow implications, repayment risks, and potential strategies to improve cash flow. Compute the Debt-to-Equity ratio and Interest Coverage ratio. Identify potential repayment risks based on available cash flow and debt obligations. Provide actionable recommendations to enhance debt management and suggest strategies to optimize and increase cash flow.

Your response must be strictly formatted as a JSON object with the following structure and no additional text:

{
  "debt_analysis": {
    "leverage_ratios": {
      "debt_to_equity": "<computed_ratio>",
      "interest_coverage": "<computed_ratio>"
    },
    "cash_flow_implications": "<impact_of_interest_expense_on_cash_flow>",
    "repayment_risks": "<analysis_of_company's_ability_to_meet_debt_obligations>",
    "recommendations": [
      "<recommendation1>",
      "<recommendation2>",
      "<recommendation3>"
    ],
    "cash_flow_suggestions": [
      "<strategy1>",
      "<strategy2>",
      "<strategy3>"
    ]
  }
}"""
        )

        user_prompt = UserMessage(financial_data)

        try:
            client = ChatCompletionsClient(
                endpoint="https://models.github.ai/inference",
                credential=AzureKeyCredential(
                    "Replace with Your github Pat token.")
            )

            response = client.complete(
                messages=[system_prompt, user_prompt],
                model="gpt-4o-mini",
                temperature=1,
                max_tokens=4096,
                top_p=1
            )

            # Extract response content (expected JSON)
            analysis_result = response.choices[0].message.content.strip()

            # **Clean the AI response (Remove unwanted formatting)**
            analysis_result = analysis_result.replace(
                '```json', '').replace('```', '').strip()
            if analysis_result.startswith('"') and analysis_result.endswith('"'):
                analysis_result = analysis_result[1:-1]

            # Try parsing cleaned JSON response
            try:
                analysis_json = json.loads(analysis_result)
            except json.JSONDecodeError:
                # Return raw output if parsing fails
                analysis_json = {"raw": analysis_result}

            return JsonResponse({"analysis": analysis_json}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed."}, status=405)


@csrf_exempt
def process_investment_strategy(request):
    # Ensure only POST method is allowed
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed."}, status=405)

    # Parse the JSON payload
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    # Validate user_query field
    user_query = data.get("user_query")
    if not user_query:
        return JsonResponse({"error": "The 'user_query' field is required."}, status=400)

    # Initialize the ChatCompletionsClient
    try:
        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential("Replace with Your github Pat token.")
        )
    except Exception as e:
        return JsonResponse({"error": f"Error initializing client: {str(e)}"}, status=500)

    # Define the system prompt for the AI model
    system_prompt = """
You are a highly intelligent and detail-oriented Investment Strategy Advisor specializing in budget-based investment planning. Your role is to analyze provided financial data and budgets to suggest appropriate asset allocations. Provide your response strictly as a JSON object with the following structure and no additional text:

```json
{
  "investment_strategy": {
    "budget": "<amount>",
    "risk_profile": "<low/medium/high>",
    "recommended_allocations": {
      "stocks": "<percentage>",
      "bonds": "<percentage>",
      "real_estate": "<percentage>",
      "gold": "<percentage>" // Additional Suggestion user Ask for. Or remove if user not Asked.
    },
    "justification": "<explanation> in more detail"
  }
}
"""
    try:
        response = client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(user_query)
            ],
            model="gpt-4o-mini",
            temperature=1,
            max_tokens=4096,
            top_p=1
        )
    except Exception as e:
        return JsonResponse({"error": f"Error from inference API: {str(e)}"}, status=500)

    answer_text = response.choices[0].message.content.strip()
    if answer_text.startswith("```json") and answer_text.endswith("```"):
        answer_text = answer_text[7:-3].strip()
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON response",
            "raw_response": answer_text
        }, status=500)

    return JsonResponse({"data": answer_json}, status=200)

# def load_market_data(request):
#     if request.method != "GET":
#         return JsonResponse({"error": "Only GET method allowed."}, status=405)
    
#     try:
#         # Update the path to your JSON file if needed.
#         with open("BSE.json", "r") as f:
#             data = json.load(f)
#     except Exception as e:
#         return JsonResponse({"error": f"Failed to load JSON file: {str(e)}"}, status=500)
    
#     inserted_count = 0
#     try:
#         # Iterate over each industry and its list of companies.
#         for industry, records in data.items():
#             for record in records:
#                 # Add the Industry field to each record.
#                 record["Industry"] = industry

#                 # Create a MarketOccupancy instance from the record.
#                 MarketOccupancy.objects.create(
#                     Industry=record.get("Industry"),
#                     fincode=record.get("fincode"),
#                     symbol=record.get("symbol"),
#                     compname=record.get("compname"),
#                     S_NAME=record.get("S_NAME"),
#                     CLOSE_PRICE=str(record.get("CLOSE_PRICE")),  # Converting to string if needed
#                     Change=str(record.get("Change")),
#                     PerChange=str(record.get("PerChange")),
#                     MCAP=str(record.get("MCAP")),
#                     PE=str(record.get("PE")),
#                     PB=str(record.get("PB"))
#                 )
#                 inserted_count += 1

#         return JsonResponse({"status": "success", "inserted": inserted_count})
#     except Exception as e:
#         return JsonResponse({"error": f"Failed to save data: {str(e)}"}, status=500)
    
# @csrf_exempt
# def delete_all_market_occupancy(request):
#     if request.method != "DELETE":
#         return JsonResponse({"error": "Only DELETE method allowed."}, status=405)
    
#     try:
#         # Delete all records in the MarketOccupancy model
#         deleted_count, _ = MarketOccupancy.objects.all().delete()
#         return JsonResponse({"status": "success", "deleted_count": deleted_count})
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# @api_view(['GET'])
# @csrf_exempt
# def list_industries(request):
#     industries = MarketOccupancy.objects.values_list('Industry', flat=True).distinct()
#     return Response({"industries": list(industries)})

@csrf_exempt
def industry_occupancy(request):
    if request.method != "GET":
        return JsonResponse({"error": "Only GET method allowed."}, status=405)

    industry_name = request.GET.get("industry")
    if not industry_name:
        return JsonResponse({"error": "Industry name is required as 'industry' parameter."}, status=400)

    qs = MarketOccupancy.objects.filter(Industry=industry_name)
    if not qs.exists():
        return JsonResponse({"error": f"No data found for industry '{industry_name}'."}, status=404)

    total_mcap = 0.0
    records = []

    for obj in qs:
        try:
            mcap_val = float(obj.MCAP.replace(',', '')) if obj.MCAP else 0.0
        except ValueError:
            mcap_val = 0.0
        total_mcap += mcap_val
        records.append({
            "fincode": obj.fincode,
            "symbol": obj.symbol,
            "compname": obj.compname,
            "S_NAME": obj.S_NAME,
            "CLOSE_PRICE": obj.CLOSE_PRICE,
            "Change": obj.Change,
            "PerChange": obj.PerChange,
            "MCAP": obj.MCAP,
            "PE": obj.PE,
            "PB": obj.PB,
        })

    for record in records:
        try:
            mcap_val = float(record["MCAP"].replace(',', '')) if record["MCAP"] else 0.0
        except ValueError:
            mcap_val = 0.0
        occupancy = (mcap_val / total_mcap * 100) if total_mcap > 0 else 0
        record["occupancy"] = round(occupancy, 2)

    # Sort the records by occupancy in descending order
    records.sort(key=lambda x: x["occupancy"], reverse=True)

    response_data = {
        "industry": industry_name,
        "total_MCAP": total_mcap,
        "data": records
    }

    return JsonResponse(response_data)