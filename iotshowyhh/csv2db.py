#coding:utf-8
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iotshowyhh.settings")
import django
import pandas as pd
if django.VERSION >= (1, 7):#自动判断版本
    django.setup()
from dianbiao.models import Timeseries
import random

DATE_PATH = "./dianbiao_lgb.csv"

def main():
    dianbiao = pd.read_csv(DATE_PATH)
    dianbiao = dianbiao.fillna(-1)
    dianbiao['Datetime'] = dianbiao['Datetime'].str.replace('\t','')
    # print(len(dianbiao.iloc[0]))
    # print(dianbiao.iloc[0])
    # print('\n\n\n')
    TimeseriesList = []
    cnt = 1
    for (num, Index, Datetime, Instantaneous_active_power, Wattless_power, A_phase_current, B_phase_current,
        C_phase_current, A_phase_voltage, B_phase_voltage, C_phase_voltage, Total_power_factor, Forward_active_power,
        One_quadrant_wattless_power, Four_quadrant_wattless_power, date, year, month, day, hour, minute,
        second, dayofyear, week, dayofweek, time_epoch, isweekend, yesterday_label, last_week_label,
        max_last3_day_label, min_last3_day_label, mean_last3_day_label, median_last3_day_label,
        var_last3_day_label, max_last7_day_label, min_last7_day_label, mean_last7_day_label, median_last7_day_label,
        var_last7_day_label, date_day) in dianbiao.itertuples():

        TimeseriesList.append(Timeseries(timeseries_time=Datetime,
                                         Instantaneous_active_power=Instantaneous_active_power,
                                         Predict_Instantaneous_active_power=float(Instantaneous_active_power)+random.random(),
                                         Wattless_power=Wattless_power,
                                         A_phase_current=A_phase_current,
                                         B_phase_current=B_phase_current,
                                         C_phase_current=C_phase_current,
                                         A_phase_voltage=A_phase_voltage,
                                         B_phase_voltage=B_phase_voltage,
                                         C_phase_voltage=C_phase_voltage,
                                         Total_power_factor=Total_power_factor,
                                         Forward_active_power=Forward_active_power,
                                         One_quadrant_wattless_power=One_quadrant_wattless_power,
                                         Four_quadrant_wattless_power=Four_quadrant_wattless_power,
                                         year=year,
                                         month=month,
                                         day=day,
                                         hour=hour,
                                         minute=minute,
                                         second=second,
                                         dayofyear=dayofyear,
                                         week=week,
                                         dayofweek=dayofweek,
                                         isweekend=isweekend,
                                         time_epoch=time_epoch,
                                         yesterday_label=yesterday_label,
                                         last_week_label=last_week_label,
                                         max_last3_day_label=max_last3_day_label,
                                         min_last3_day_label=min_last3_day_label,
                                         mean_last3_day_label=mean_last3_day_label,
                                         median_last3_day_label=median_last3_day_label,
                                         var_last3_day_label=var_last3_day_label,
                                         max_last7_day_label=max_last7_day_label,
                                         min_last7_day_label=min_last7_day_label,
                                         mean_last7_day_label=mean_last7_day_label,
                                         median_last7_day_label=median_last7_day_label,
                                         var_last7_day_label=var_last7_day_label,
                                         date_day=date_day))
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
    Timeseries.objects.bulk_create(TimeseriesList)

if __name__ == "__main__":
    main()
    print('Done!')