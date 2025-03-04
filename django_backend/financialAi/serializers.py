from rest_framework import serializers
from .models import FinancialAI, MarketOccupancy


class FinancialAISerializer(serializers.ModelSerializer):
    class Meta:
        model = FinancialAI
        fields = "__all__"

class MarketOccupancySerializers(serializers.ModelSerializer):
    class Meta:
        model = MarketOccupancy
        fields = "__all__"
