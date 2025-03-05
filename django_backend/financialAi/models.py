from django.db import models
from django.utils import timezone
# Create your models here.
class FinancialAI(models.Model):
    user_query = models.TextField(blank=True, null=True)
    AI_Response = models.JSONField(default=list, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Extracted Data for {self.user_query}"

class MarketOccupancy(models.Model):
    Industry = models.CharField(blank=True, null=True, max_length=255)
    fincode = models.CharField(blank=True, null=True, max_length=255)
    symbol = models.CharField(blank=True, null=True, max_length=255)
    compname = models.CharField(blank=True, null=True, max_length=255)
    S_NAME = models.CharField(blank=True, null=True, max_length=255)
    CLOSE_PRICE = models.CharField(blank=True, null=True, max_length=255)
    Change = models.CharField(blank=True, null=True, max_length=255)
    PerChange = models.CharField(blank=True, null=True, max_length=255)
    MCAP = models.CharField(blank=True, null=True, max_length=255)
    PE = models.CharField(blank=True, null=True, max_length=255)
    PB = models.CharField(blank=True, null=True, max_length=255)

    def __str__(self):
        return f"{self.compname} and Market Cap {self.MCAP}"