from django.urls import path
from . import views
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'Financial_ai_data', views.FinancialAIViewset)
router.register(r'MarketOccupancySerializers', views.MarketOccupancySerializers)

urlpatterns = [
    path('process_financial_query/', views.process_financial_query, name='process_financial_query'),
    path('predict/', views.predict_stock, name='predict'),
    path('pre_debt/', views.analyze_debt, name='analyze_debt'),
    path('investment_pre/', views.process_investment_strategy, name='process_investment_strategy'),
    # path('load_market_data/', views.load_market_data, name="load_market_data"),
    #  path('delete/', views.delete_all_market_occupancy, name='delete_marketoccupancy'),
        # path('industries/', views.list_industries, name='list-industries'),
    path('industry-occupancy/', views.industry_occupancy, name='industry_occupancy'),
]

urlpatterns += router.urls