from django.db import models


class timeseries(models.Model):
    timeseries_time = models.CharField(max_length=20)
    timeseries_Instantaneous_active_power = models.CharField(max_length=20)
    timeseries_Wattless_power = models.CharField(max_length=20)
    timeseries_A_phase_voltage = models.CharField(max_length=20)
    timeseries_B_phase_voltage = models.CharField(max_length=20)
    timeseries_C_phase_voltage = models.CharField(max_length=20)
    timeseries_A_phase_current = models.CharField(max_length=20)
    timeseries_B_phase_current = models.CharField(max_length=20)
    timeseries_C_phase_current = models.CharField(max_length=20)
    timeseries_Total_power_factor = models.CharField(max_length=20)
    timeseries_Forward_active_power = models.CharField(max_length=20)
    timeseries_One_quadrant_wattless_power = models.CharField(max_length=20)
    timeseries_Four_quadrant_wattless_power = models.CharField(max_length=20)
    timeseries_year = models.CharField(max_length=20)
    timeseries_month = models.CharField(max_length=20)
    timeseries_day = models.CharField(max_length=20)
    timeseries_hour = models.CharField(max_length=20)
    timeseries_minute = models.CharField(max_length=20)
    timeseries_second = models.CharField(max_length=20)
    timeseries_dayofyear = models.CharField(max_length=20)
    timeseries_week = models.CharField(max_length=20)
    timeseries_dayofweek = models.CharField(max_length=20)
    timeseries_isweekend = models.CharField(max_length=20)
    timeseries_time_epoch = models.CharField(max_length=20)
    
# Create your models here.
