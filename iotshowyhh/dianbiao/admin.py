from django.contrib import admin
from .models import Timeseries

class TimeseriesAdmin(admin.ModelAdmin):
    fields = ['timeseries_time', 'Instantaneous_active_power', 'Predict_Instantaneous_active_power', 'Wattless_power',
       'A_phase_current', 'B_phase_current', 'C_phase_current', 'A_phase_voltage', 'B_phase_voltage', 'C_phase_voltage',
       'Total_power_factor', 'Forward_active_power', 'One_quadrant_wattless_power', 'Four_quadrant_wattless_power',
       'year', 'month', 'day', 'hour', 'minute', 'second', 'dayofyear', 'week',
       'dayofweek', 'time_epoch', 'isweekend', 'yesterday_label', 'last_week_label', 'max_last3_day_label',
       'min_last3_day_label', 'mean_last3_day_label', 'median_last3_day_label', 'var_last3_day_label',
       'max_last7_day_label', 'min_last7_day_label', 'mean_last7_day_label', 'median_last7_day_label',
       'var_last7_day_label','date_day']

admin.site.register(Timeseries, TimeseriesAdmin)
# Register your models here.
