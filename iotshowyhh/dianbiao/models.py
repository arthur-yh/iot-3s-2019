from django.db import models
import datetime

class Timeseries(models.Model):
    timeseries_time = models.DateTimeField()
    Instantaneous_active_power = models.FloatField()
    Predict_Instantaneous_active_power = models.FloatField()
    Wattless_power = models.FloatField()
    A_phase_current = models.FloatField()
    B_phase_current = models.FloatField()
    C_phase_current = models.FloatField()
    A_phase_voltage = models.FloatField()
    B_phase_voltage = models.FloatField()
    C_phase_voltage = models.FloatField()
    Total_power_factor = models.FloatField()
    Forward_active_power = models.FloatField()
    One_quadrant_wattless_power = models.FloatField()
    Four_quadrant_wattless_power = models.FloatField()
    year = models.FloatField(default=1)
    month = models.FloatField(default=1)
    day = models.FloatField(default=1)
    hour = models.FloatField(default=1)
    minute = models.FloatField(default=1)
    second = models.FloatField(default=1)
    dayofyear = models.FloatField(default=1)
    week = models.FloatField(default=1)
    dayofweek = models.FloatField(default=1)
    isweekend = models.FloatField(default=1)
    time_epoch = models.FloatField(default=1)
    yesterday_label = models.FloatField(default=1)
    last_week_label = models.FloatField(default=1)
    max_last3_day_label = models.FloatField(default=1)
    min_last3_day_label = models.FloatField(default=1)
    mean_last3_day_label = models.FloatField(default=1)
    median_last3_day_label = models.FloatField(default=1)
    var_last3_day_label = models.FloatField(default=1)
    max_last7_day_label = models.FloatField(default=1)
    min_last7_day_label = models.FloatField(default=1)
    mean_last7_day_label = models.FloatField(default=1)
    median_last7_day_label = models.FloatField(default=1)
    var_last7_day_label = models.FloatField(default=1)
    date_day = models.DateField(default=datetime.datetime.now().date())

    def __str__(self):
        return str(self.timeseries_time)

# Create your models here.
