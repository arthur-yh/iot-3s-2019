import datetime
import json
import numpy as np
import os
import lightgbm

from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, HttpResponse
from django.db.models import Sum
from django.views.decorators.csrf import csrf_exempt, csrf_protect

from sklearn.externals import joblib
from .models import Timeseries

MODEL_PATH = "dianbiao/ML_models/"
ORI_DATA_PATH = "dianbiao_lgb.csv"
current_date=datetime.datetime(2017,6,6,6,0)
shengchan_name="江苏省苏州工业园区133号"
shengchan_type="饮品加工"
jiagonggongyi="溶糖"
jiancedian="一个"
yunzhuan="正常"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def date_is_valid(current_date):
    if type(current_date) == datetime.datetime:
        cdate = current_date
    else:
        cdate = datetime.datetime.strptime(current_date, '%Y-%m-%dT%H:%M')
    if (cdate.year in [2016, 2017]) & (cdate.minute % 15 == 0):
        return True
    else:
        return False

def load_models():
    lgb_model = joblib.load(MODEL_PATH + "lgb.pkl")
    guize_A = joblib.load(MODEL_PATH + "regr_A.pkl")
    guize_C = joblib.load(MODEL_PATH + "regr_C.pkl")
    scaler = joblib.load(MODEL_PATH + "lstm_standerd")
    # LSTM_model = joblib.load(MODEL_PATH + "lstm.pkl")
    return lgb_model, guize_A, guize_C

def get_current_date(request):
    """ 获取用户输入信息"""
    return render(request, 'dianbiao/get_current_date.html')

def control(request):
    """ 电表控制信息"""
    return render(request, 'dianbiao/control.html')

def alarm(request):
    """ 数据警报系统"""
    if type(current_date) == datetime.datetime:
        cdate = current_date
    else:
        cdate = datetime.datetime.strptime(current_date, '%Y-%m-%dT%H:%M')
    show_time = str(cdate.year) + str('.') + str(cdate.month) + str('.') + str(cdate.day) + str(' ') + \
                str(cdate.hour) + str(':') + str(cdate.minute)
    show_time_now = str(cdate.hour) + str(':') + str(cdate.minute)
    show_time_last15 = str((cdate - datetime.timedelta(minutes=15)).hour) + str(':') + str((cdate - datetime.timedelta(minutes=15)).minute)
    show_time_last30 = str((cdate - datetime.timedelta(minutes=30)).hour) + str(':') + str(
        (cdate - datetime.timedelta(minutes=30)).minute)
    show_time_last45 = str((cdate - datetime.timedelta(minutes=45)).hour) + str(':') + str(
        (cdate - datetime.timedelta(minutes=45)).minute)
    cdate_min = cdate - datetime.timedelta(days=1)
    Current_timeseries_list = Timeseries.objects.filter(timeseries_time__gte=cdate_min). \
        filter(timeseries_time__lte=cdate)
    # 获取当前天的功率因数数据
    current_factor_data = {}
    # print(Current_timeseries_list)
    for index, Timeseries_objects in enumerate(Current_timeseries_list):
        # print(Timeseries_objects.timeseries_time)
        keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
               + str(Timeseries_objects.timeseries_time.minute)
        current_factor_data[keys] = Timeseries_objects.Total_power_factor
        if index == (len(Current_timeseries_list)-1):
            current_factor = Timeseries_objects.Total_power_factor
    if current_factor>=0.9:
        factor_color = "#4dcbb5"
        factor_alarm = "当前功率因数正常"
    elif current_factor>0.8:
        factor_color = "#fd6265"
        factor_alarm = "当前功率因数在0.8-0.9之间，无功功率变大，请检查是否有异常情况"
    else:
        factor_color = "#fd6265"
        factor_alarm = "当前功率因数已小于0.8，将产生大量调度电费，电网会不稳定！"
    # 获取当前时刻以及前4个时刻的瞬时功率以及过去14天这个时刻的瞬时有功功率分布情况
    last_hour_time = cdate - datetime.timedelta(hours=1)
    last_hour_time_list = Timeseries.objects.filter(timeseries_time__gte=last_hour_time). \
        filter(timeseries_time__lte=cdate)
    real_active_power = {}
    last14_days_active_power = {}
    label_keys = []
    for index, Timeseries_objects in enumerate(last_hour_time_list):
        keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
               + str(Timeseries_objects.timeseries_time.minute)
        label_keys.append(keys)
        last14_days_active_power[keys] = []
        real_active_power[keys] = Timeseries_objects.Instantaneous_active_power
        if index == (len(last_hour_time_list)-1):
            current_active_power = Timeseries_objects.Instantaneous_active_power
    for i in range(1, 31, 1):
        cdate_temp = cdate - datetime.timedelta(days=i)
        cdate_last_hour = cdate_temp - datetime.timedelta(hours=1)
        temp_Timeserieslist = Timeseries.objects.filter(timeseries_time__gte=cdate_last_hour). \
            filter(timeseries_time__lte=cdate_temp)
        for index, Timeseries_objects in enumerate(temp_Timeserieslist):
            last14_days_active_power[label_keys[index]].append(Timeseries_objects.Instantaneous_active_power)
    max_last14_active_power = max(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    min_last14_active_power = min(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    mean_last14_active_power = np.mean(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    median_last14_active_power = np.median(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    bianhua_active_power = round((current_active_power - mean_last14_active_power) / mean_last14_active_power*100, 2)
    max_last14_active_power = "max:    " + str(max_last14_active_power)
    min_last14_active_power = "min:    " + str(min_last14_active_power)
    mean_last14_active_power = "mean:   " + str(round(mean_last14_active_power, 2))
    median_last14_active_power = "median: " + str(round(median_last14_active_power,2))
    if bianhua_active_power >= 30:
        active_alarm = "当前有功功率增幅超过30%，疑似有盗电等异常情况"
    elif bianhua_active_power >= 15:
        active_alarm = "当前有功功率增幅超过15%，电力消耗增幅过大，应当予以重视"
    elif bianhua_active_power >= 10:
        active_alarm = "当前有功功率增幅超过10%，近期是否有大功率机器加入"
    elif bianhua_active_power >= -10:
        active_alarm = "当前有功功率幅度变化属于正常情况"
    elif bianhua_active_power >= -15:
        active_alarm = "当前有功功率幅度降低超过10%-15%，是否有机器故障需要"
    else:
        active_alarm = "当前有功功率幅度降低超过15%，需要排除系统停机故障"
    if bianhua_active_power >= 0:
        bianhua_active_power = str('+') + str(bianhua_active_power) + str('%')
        bianhua_active_power_color = "#fd6265"
    else:
        bianhua_active_power = str(bianhua_active_power) + str('%')
        bianhua_active_power_color = "#4dcbb5"
    context = {'show_time': show_time,
               'show_time_now': show_time_now,
               'show_time_last15': show_time_last15,
               'show_time_last30': show_time_last30,
               'show_time_last45': show_time_last45,
               'shengchan_name': shengchan_name,
               'shengchan_type': shengchan_type,
               'yunzhuan': yunzhuan,
               'jiagonggongyi': jiagonggongyi,
               'jiancedian': jiancedian,
               'current_factor': current_factor,
               'current_factor_data': json.dumps(current_factor_data),
               'factor_color': factor_color,
               'factor_alarm': factor_alarm,
               'current_active_power': current_active_power,
               'max_last14_active_power': max_last14_active_power,
               'min_last14_active_power': min_last14_active_power,
               'mean_last14_active_power': mean_last14_active_power,
               'median_last14_active_power': median_last14_active_power,
               'bianhua_active_power': bianhua_active_power,
               'bianhua_active_power_color': bianhua_active_power_color,
               'active_alarm': active_alarm,
               'real_active_power': json.dumps(real_active_power),
               'last30_days_active_power_one': json.dumps(last14_days_active_power[label_keys[0]]),
               'last30_days_active_power_two': json.dumps(last14_days_active_power[label_keys[1]]),
               'last30_days_active_power_three': json.dumps(last14_days_active_power[label_keys[2]]),
               'last30_days_active_power_four': json.dumps(last14_days_active_power[label_keys[3]])
               }

    return render(request, 'dianbiao/effiency.html', context)

@csrf_exempt
def refresh_alarm(request):
    print('********************************* 开始更新alarm *********************************')
    cdate = request.POST.get('current_date')
    cdate = datetime.datetime.strptime(cdate, '%Y.%m.%d %H:%M')
    cdate = cdate + datetime.timedelta(minutes=15)
    print('********************************* 当前更新alarm时间%s *********************************' % cdate)
    show_time = str(cdate.year) + str('.') + str(cdate.month) + str('.') + str(cdate.day) + str(' ') + \
                str(cdate.hour) + str(':') + str(cdate.minute)
    show_time_now = str(cdate.hour) + str(':') + str(cdate.minute)
    show_time_last15 = str((cdate - datetime.timedelta(minutes=15)).hour) + str(':') + str((cdate - datetime.timedelta(minutes=15)).minute)
    show_time_last30 = str((cdate - datetime.timedelta(minutes=30)).hour) + str(':') + str(
        (cdate - datetime.timedelta(minutes=30)).minute)
    show_time_last45 = str((cdate - datetime.timedelta(minutes=45)).hour) + str(':') + str(
        (cdate - datetime.timedelta(minutes=45)).minute)
    cdate_min = cdate - datetime.timedelta(days=1)
    Current_timeseries_list = Timeseries.objects.filter(timeseries_time__gte=cdate_min). \
        filter(timeseries_time__lte=cdate)
    # 获取当前天的功率因数数据
    current_factor_data = {}
    # print(Current_timeseries_list)
    for index, Timeseries_objects in enumerate(Current_timeseries_list):
        # print(Timeseries_objects.timeseries_time)
        keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
               + str(Timeseries_objects.timeseries_time.minute)
        current_factor_data[keys] = Timeseries_objects.Total_power_factor
        if index == (len(Current_timeseries_list) - 1):
            current_factor = Timeseries_objects.Total_power_factor
    if current_factor >= 0.9:
        factor_color = "#4dcbb5"
        factor_alarm = "当前功率因数正常"
    elif current_factor > 0.8:
        factor_color = "#fd6265"
        factor_alarm = "当前功率因数在0.8-0.9之间，无功功率变大，请检查是否有异常情况"
    else:
        factor_color = "#fd6265"
        factor_alarm = "当前功率因数已小于0.8，将产生大量调度电费，电网会不稳定！"
    # 获取当前时刻以及前4个时刻的瞬时功率以及过去14天这个时刻的瞬时有功功率分布情况
    last_hour_time = cdate - datetime.timedelta(hours=1)
    last_hour_time_list = Timeseries.objects.filter(timeseries_time__gte=last_hour_time). \
        filter(timeseries_time__lte=cdate)
    real_active_power = {}
    last14_days_active_power = {}
    label_keys = []
    for index, Timeseries_objects in enumerate(last_hour_time_list):
        keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
               + str(Timeseries_objects.timeseries_time.minute)
        label_keys.append(keys)
        last14_days_active_power[keys] = []
        real_active_power[keys] = Timeseries_objects.Instantaneous_active_power
        # print(index, len(last_hour_time_list))
        if index == (len(last_hour_time_list) - 1):
            current_active_power = Timeseries_objects.Instantaneous_active_power
    for i in range(1, 31, 1):
        cdate_temp = cdate - datetime.timedelta(days=i)
        cdate_last_hour = cdate_temp - datetime.timedelta(hours=1)
        temp_Timeserieslist = Timeseries.objects.filter(timeseries_time__gte=cdate_last_hour). \
            filter(timeseries_time__lte=cdate_temp)
        for index, Timeseries_objects in enumerate(temp_Timeserieslist):
            last14_days_active_power[label_keys[index]].append(Timeseries_objects.Instantaneous_active_power)
    max_last14_active_power = max(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    min_last14_active_power = min(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    mean_last14_active_power = np.mean(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    median_last14_active_power = np.median(last14_days_active_power[str(cdate.hour) + str(':') + str(cdate.minute)])
    bianhua_active_power = round((current_active_power - mean_last14_active_power) / mean_last14_active_power * 100,
                                 2)
    max_last14_active_power = "max:    " + str(max_last14_active_power)
    min_last14_active_power = "min:    " + str(min_last14_active_power)
    mean_last14_active_power = "mean:   " + str(round(mean_last14_active_power, 2))
    median_last14_active_power = "median: " + str(round(median_last14_active_power,2))
    if bianhua_active_power >= 30:
        active_alarm = "当前有功功率增幅超过30%，疑似有盗电等异常情况"
    elif bianhua_active_power >= 15:
        active_alarm = "当前有功功率增幅超过15%，电力消耗增幅过大，应当予以重视"
    elif bianhua_active_power >= 10:
        active_alarm = "当前有功功率增幅超过10%，近期是否有大功率机器加入"
    elif bianhua_active_power >= -10:
        active_alarm = "当前有功功率幅度变化属于正常情况"
    elif bianhua_active_power >= -15:
        active_alarm = "当前有功功率幅度降低超过10%-15%，是否有机器故障需要"
    else:
        active_alarm = "当前有功功率幅度降低超过15%，需要排除系统停机故障"
    if bianhua_active_power >= 0:
        bianhua_active_power = str('+') + str(bianhua_active_power) + str('%')
        bianhua_active_power_color = "#fd6265"
    else:
        bianhua_active_power = str(bianhua_active_power) + str('%')
        bianhua_active_power_color = "#4dcbb5"
    # 设置返回参数
    response = HttpResponse()
    text_info_alarm = {}
    text_info_alarm['show_time'] = show_time
    text_info_alarm['show_time_now'] = show_time_now
    text_info_alarm['show_time_last15'] = show_time_last15
    text_info_alarm['show_time_last30'] = show_time_last30
    text_info_alarm['show_time_last45'] = show_time_last45
    text_info_alarm['current_factor'] = current_factor
    text_info_alarm['factor_color'] = factor_color
    text_info_alarm['factor_alarm'] = factor_alarm
    text_info_alarm['current_active_power'] = current_active_power
    text_info_alarm['max_last14_active_power'] = max_last14_active_power
    text_info_alarm['min_last14_active_power'] = min_last14_active_power
    text_info_alarm['mean_last14_active_power'] = mean_last14_active_power
    text_info_alarm['median_last14_active_power'] = median_last14_active_power
    text_info_alarm['bianhua_active_power'] = bianhua_active_power
    text_info_alarm['bianhua_active_power_color'] = bianhua_active_power_color
    text_info_alarm['active_alarm'] = active_alarm

    for i, key in enumerate(real_active_power.keys()):
        text_info_alarm["data"+str(i)] = real_active_power[key]
        # print(text_info_alarm["data"+str(i)])

    response.write('refresh_alarm='+json.dumps({'text_info_alarm': text_info_alarm, 'real_active_power': real_active_power, \
                                                'current_factor_data': current_factor_data,
                                                'last30_days_active_power_one': last14_days_active_power[label_keys[0]], \
                                                'last30_days_active_power_two': last14_days_active_power[label_keys[1]],\
                                                'last30_days_active_power_three': last14_days_active_power[label_keys[2]], \
                                                'last30_days_active_power_four': last14_days_active_power[label_keys[3]]}))

    response['Content-Type'] = "text/javascript"

    return response

@csrf_exempt
def index(request):
    """根据输入信息进行处理并返回渲染后的html界面"""
    global current_date
    global shengchan_name
    global shengchan_type
    global jiagonggongyi
    global jiancedian
    global yunzhuan
    try:
        current_date = request.POST["a"]
        shengchan_name = request.POST["shengchan_name"]
        shengchan_type = request.POST.get("shengchan_type")
        jiagonggongyi = request.POST.get("jiagonggongyi")
        jiancedian = request.POST.get("jiancedian")
        yunzhuan = request.POST.get("yunzhuan")
    except:
        pass

    if not date_is_valid(current_date):
        return HttpResponse('当前日期没有数据')
    else:
        # 载入ML模型
        if yunzhuan != str('正常'):
            yunzhuan_color = "#fd6265"
        else:
            yunzhuan_color = "#4dcbb5"
        lgb_model, guize_A, guize_C = load_models()
        cdate = datetime.datetime.strptime(current_date, '%Y-%m-%dT%H:%M')
        show_time = str(cdate.year) + str('.') + str(cdate.month) + str('.') + str(cdate.day) + str(' ') + \
                    str(cdate.hour) + str(':') + str(cdate.minute)
        cdate_min = cdate - datetime.timedelta(days=1)
        cdate_max = cdate + datetime.timedelta(days=1)
        Current_timeseries_list = Timeseries.objects.filter(timeseries_time__gte=cdate_min).\
            filter(timeseries_time__lte=cdate)
        Predict_timeseries_list = Timeseries.objects.filter(timeseries_time__gte=cdate_min).\
            filter(timeseries_time__lte=cdate_max)
        current_data = {}
        current_predict_data = {}
        predict_data = {}
        # LSTM 预测分析
        USE_lstm = False
        last2_day_timeseries_list = Timeseries.objects.filter(date_day=(cdate - datetime.timedelta(days=2)).date())
        last_day_timeseries_list = Timeseries.objects.filter(date_day=cdate_min.date())
        now_timeseries_list = Timeseries.objects.filter(date_day=cdate.date())
        # 产生一天24个小时共96个点的keys
        keys_96_points = {}
        cnt = 0
        for i in range(0, 24):
            for j in [':0', ':15', ':30', ':45']:
                value = str(i) + j
                keys_96_points[cnt] = value
                cnt += 1
        # 如果预测的3天内有一天没有标准的96的预测数据，那么就不用lstm
        # Get current data for current day
        for Timeseries_objects in Current_timeseries_list:
            # print(Timeseries_objects.timeseries_time)
            keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
                   + str(Timeseries_objects.timeseries_time.minute)
            current_data[keys] = Timeseries_objects.Instantaneous_active_power
            guize_predict = 0.6 * guize_A.predict(np.array([Timeseries_objects.A_phase_current]).reshape(-1, 1)) + \
                            0.4 * guize_C.predict(np.array([Timeseries_objects.C_phase_current]).reshape(-1, 1))
            # print('guize_predict:', guize_predict[0][0])
            # lgb预测
            lgb_array = np.array([[Timeseries_objects.Wattless_power,
                                  Timeseries_objects.A_phase_current,
                                  Timeseries_objects.B_phase_current,
                                  Timeseries_objects.C_phase_current,
                                  Timeseries_objects.A_phase_voltage,
                                  Timeseries_objects.B_phase_voltage,
                                  Timeseries_objects.C_phase_voltage,
                                  Timeseries_objects.Total_power_factor,
                                  Timeseries_objects.Forward_active_power,
                                  Timeseries_objects.One_quadrant_wattless_power,
                                  Timeseries_objects.Four_quadrant_wattless_power,
                                  Timeseries_objects.year,
                                  Timeseries_objects.month,
                                  Timeseries_objects.day,
                                  Timeseries_objects.hour,
                                  Timeseries_objects.minute,
                                  Timeseries_objects.second,
                                  Timeseries_objects.dayofyear,
                                  Timeseries_objects.week,
                                  Timeseries_objects.dayofweek,
                                  Timeseries_objects.time_epoch,
                                  Timeseries_objects.isweekend,
                                  Timeseries_objects.yesterday_label,
                                  Timeseries_objects.last_week_label,
                                  Timeseries_objects.max_last3_day_label,
                                  Timeseries_objects.min_last3_day_label,
                                  Timeseries_objects.mean_last3_day_label,
                                  Timeseries_objects.median_last3_day_label,
                                  Timeseries_objects.var_last3_day_label,
                                  Timeseries_objects.max_last7_day_label,
                                  Timeseries_objects.min_last7_day_label,
                                  Timeseries_objects.mean_last7_day_label,
                                  Timeseries_objects.median_last7_day_label,
                                  Timeseries_objects.var_last7_day_label]]).reshape(1, -1)
            # print(lgb_array.shape)
            lgb_predict = np.expm1(lgb_model.predict(lgb_array))
            # print(lgb_predict)
            # 将lstm预测的数据整合到模型当中
            current_predict_data[keys] = 0.6 * lgb_predict[0] + 0.4 * guize_predict[0][0]
        # Get predict for 2 days
        for Timeseries_objects in Predict_timeseries_list:
            keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
                   + str(Timeseries_objects.timeseries_time.minute)
            guize_predict = 0.6 * guize_A.predict(np.array([Timeseries_objects.A_phase_current]).reshape(-1, 1)) + \
                            0.4 * guize_C.predict(np.array([Timeseries_objects.C_phase_current]).reshape(-1, 1))
            # lgb预测
            lgb_array = np.array([[Timeseries_objects.Wattless_power,
                                  Timeseries_objects.A_phase_current,
                                  Timeseries_objects.B_phase_current,
                                  Timeseries_objects.C_phase_current,
                                  Timeseries_objects.A_phase_voltage,
                                  Timeseries_objects.B_phase_voltage,
                                  Timeseries_objects.C_phase_voltage,
                                  Timeseries_objects.Total_power_factor,
                                  Timeseries_objects.Forward_active_power,
                                  Timeseries_objects.One_quadrant_wattless_power,
                                  Timeseries_objects.Four_quadrant_wattless_power,
                                  Timeseries_objects.year,
                                  Timeseries_objects.month,
                                  Timeseries_objects.day,
                                  Timeseries_objects.hour,
                                  Timeseries_objects.minute,
                                  Timeseries_objects.second,
                                  Timeseries_objects.dayofyear,
                                  Timeseries_objects.week,
                                  Timeseries_objects.dayofweek,
                                  Timeseries_objects.time_epoch,
                                  Timeseries_objects.isweekend,
                                  Timeseries_objects.yesterday_label,
                                  Timeseries_objects.last_week_label,
                                  Timeseries_objects.max_last3_day_label,
                                  Timeseries_objects.min_last3_day_label,
                                  Timeseries_objects.mean_last3_day_label,
                                  Timeseries_objects.median_last3_day_label,
                                  Timeseries_objects.var_last3_day_label,
                                  Timeseries_objects.max_last7_day_label,
                                  Timeseries_objects.min_last7_day_label,
                                  Timeseries_objects.mean_last7_day_label,
                                  Timeseries_objects.median_last7_day_label,
                                  Timeseries_objects.var_last7_day_label]]).reshape(1, -1)
            lgb_predict = np.expm1(lgb_model.predict(lgb_array))
            # 将lstm预测的数据整合到模型当中
            predict_data[keys] = 0.6 * lgb_predict[0] + 0.4 * guize_predict[0][0]
        # 计算当月截止目前的累计用电信息
        current_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year).\
            filter(timeseries_time__month=cdate.month).filter(timeseries_time__lte=cdate)\
            .all().aggregate(Sum('Instantaneous_active_power'))
        # print(current_month_ele_all)
        # 计算上个月的全部用电
        if cdate.month != 1:
            last_month = cdate.month - 1
            last_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year).\
                filter(timeseries_time__month=last_month).all().aggregate(Sum('Instantaneous_active_power'))
        else:
            last_month = 12
            last_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=12).all().aggregate(Sum('Instantaneous_active_power'))
        # 计算上个月截止到当前日的全部用电
        if cdate.month != 1:
            last_month_ele_filter_all = Timeseries.objects.filter(timeseries_time__year=cdate.year).\
                filter(timeseries_time__month=(cdate.month - 1)).filter(timeseries_time__day__lte=cdate.day)\
                .all().aggregate(Sum('Instantaneous_active_power'))
        else:
            last_month_ele_filter_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=12).all().filter(timeseries_time__day__lte=cdate.day)\
                .aggregate(Sum('Instantaneous_active_power'))
        # 计算上上个月的全部用电
        if cdate.month not in [1, 2]:
            last2_month = cdate.month - 2
            last2_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
                filter(timeseries_time__month=(cdate.month - 2)).all().aggregate(Sum('Instantaneous_active_power'))
        elif cdate.month == 2:
            last2_month = 12
            last2_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=12).all().aggregate(Sum('Instantaneous_active_power'))
        else:
            last2_month = 11
            last2_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=11).all().aggregate(Sum('Instantaneous_active_power'))
        # 计算上上上个月的全部用电
        if cdate.month not in [1, 2, 3]:
            last3_month = cdate.month - 3
            last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
                filter(timeseries_time__month=(cdate.month - 3)).all().aggregate(Sum('Instantaneous_active_power'))
        elif cdate.month == 3:
            last3_month = 12
            last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=12).all().aggregate(Sum('Instantaneous_active_power'))
        elif cdate.month == 2:
            last3_month = 11
            last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=11).all().aggregate(Sum('Instantaneous_active_power'))
        else:
            last3_month = 10
            last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
                filter(timeseries_time__month=10).all().aggregate(Sum('Instantaneous_active_power'))
        current_month_ele_all['Instantaneous_active_power__sum'] = \
            current_month_ele_all['Instantaneous_active_power__sum'] * 0.25
        last_month_ele_all['Instantaneous_active_power__sum'] = \
            last_month_ele_all['Instantaneous_active_power__sum'] * 0.25
        last_month_ele_filter_all['Instantaneous_active_power__sum'] = \
            last_month_ele_filter_all['Instantaneous_active_power__sum'] * 0.25
        last2_month_ele_all['Instantaneous_active_power__sum'] = \
            last2_month_ele_all['Instantaneous_active_power__sum'] * 0.25
        last3_month_ele_all['Instantaneous_active_power__sum'] = \
            last3_month_ele_all['Instantaneous_active_power__sum'] * 0.25
        huanbi = round((current_month_ele_all['Instantaneous_active_power__sum'] -
                  last_month_ele_filter_all['Instantaneous_active_power__sum']) /
                 last_month_ele_filter_all['Instantaneous_active_power__sum'] * 100, 2)
        if huanbi >= 0:
            huanbi_text = str('+') + str(huanbi) + str('%')
            huanbi_color_text = "#fd6265"
        else:
            huanbi_text = str(huanbi) + str('%')
            huanbi_color_text = "#4dcbb5"
        ele_fees = {}
        ele_fees[str(last3_month)] = round(last3_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
        ele_fees[str(last2_month)] = round(last2_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
        ele_fees[str(last_month)] = round(last_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
        ele_fees[str(cdate.month)] = round(current_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
        ele_fees_date = str(cdate.year) + str('.') + str(last3_month) + str('.1 - ') + str(cdate.year) + str('.') + \
                        str(cdate.month) + str('.') + str(cdate.day)
        # 用电检测部分数据
        current_month_ele_dict = {}
        for i in range(1, cdate.day):
            current_month_ele = Timeseries.objects.filter(timeseries_time__year=cdate.year).\
                filter(timeseries_time__month=cdate.month).filter(timeseries_time__day=i).\
                aggregate(Sum('Instantaneous_active_power'))
            current_month_ele_dict[str(i)] = current_month_ele['Instantaneous_active_power__sum']
        last_month_ele_dict = {}
        for i in range(1, 31):
            last_month_ele = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
                filter(timeseries_time__month=(cdate.month-1)).filter(timeseries_time__day=i). \
                aggregate(Sum('Instantaneous_active_power'))
            last_month_ele_dict[str(i)] = last_month_ele['Instantaneous_active_power__sum']
        current_month_fees_date = str(cdate.year) + str('.') + str(cdate.month) + str('.1 - ') + str(cdate.year) + str('.') + \
                        str(cdate.month) + str('.') + str(cdate.day)
        context = {'show_time': show_time,
                   'current_data': json.dumps(current_data),
                   'current_predict_data': json.dumps(current_predict_data),
                   'predict_data': json.dumps(predict_data),
                   'ele_fees': json.dumps(ele_fees),
                   'current_month_ele_dict': json.dumps(current_month_ele_dict),
                   'last_month_ele_dictl': json.dumps(last_month_ele_dict),
                   'ele_fees_date': ele_fees_date,
                   'current_month_fees_date': current_month_fees_date,
                   'date_input': cdate,
                   'yunzhuan_color': yunzhuan_color,
                   'shengchan_name': shengchan_name,
                   'shengchan_type': shengchan_type,
                   'yunzhuan': yunzhuan,
                   'jiagonggongyi': jiagonggongyi,
                   'jiancedian': jiancedian,
                   'current_month_ele_all': round(current_month_ele_all['Instantaneous_active_power__sum'], 2),
                   'current_month_ele_fees': round(current_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2),
                   'last_month_ele_all': last_month_ele_all['Instantaneous_active_power__sum'],
                   'last_month_ele_all_fees': round(last_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2),
                   'last_month_ele_filter_all': last_month_ele_filter_all['Instantaneous_active_power__sum'],
                   'last2_month_ele_all': last2_month_ele_all['Instantaneous_active_power__sum'],
                   'last2_month_ele_all_fees': round(last2_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2),
                   'last3_month_ele_all': last3_month_ele_all['Instantaneous_active_power__sum'],
                   'last3_month_ele_all_fees': round(last3_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2),
                   'huanbi_text': huanbi_text,
                   'huanbi_color_text': huanbi_color_text}
        return render(request, 'dianbiao/index.html', context)

@csrf_exempt
def refresh(request):
    print('********************************* 开始更新 *********************************')
    cdate = request.POST.get('current_date')
    cdate = datetime.datetime.strptime(cdate, '%Y.%m.%d %H:%M')
    cdate = cdate + datetime.timedelta(minutes = 15)
    print('********************************* 当前时间%s *********************************' % cdate)
    show_time = str(cdate.year) + str('.') + str(cdate.month) + str('.') + str(cdate.day) + str(' ') + \
                    str(cdate.hour) + str(':') + str(cdate.minute)
    lgb_model, guize_A, guize_C = load_models()
    cdate_min = cdate - datetime.timedelta(days=1)
    cdate_max = cdate + datetime.timedelta(days=1)
    Current_timeseries_list = Timeseries.objects.filter(timeseries_time__gte=cdate_min).\
        filter(timeseries_time__lte=cdate)
    Predict_timeseries_list = Timeseries.objects.filter(timeseries_time__gte=cdate_min).\
        filter(timeseries_time__lte=cdate_max)
    # 创建空字典存储数据
    current_data = {}
    current_predict_data = {}
    predict_data = {}
    # Get current data for current day
    for Timeseries_objects in Current_timeseries_list:
        # print(Timeseries_objects.timeseries_time)
        keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
               + str(Timeseries_objects.timeseries_time.minute)
        current_data[keys] = Timeseries_objects.Instantaneous_active_power
        guize_predict = 0.6 * guize_A.predict(np.array([Timeseries_objects.A_phase_current]).reshape(-1, 1)) + \
                        0.4 * guize_C.predict(np.array([Timeseries_objects.C_phase_current]).reshape(-1, 1))
        # print('guize_predict:', guize_predict[0][0])
        # lgb预测
        lgb_array = np.array([[Timeseries_objects.Wattless_power,
                              Timeseries_objects.A_phase_current,
                              Timeseries_objects.B_phase_current,
                              Timeseries_objects.C_phase_current,
                              Timeseries_objects.A_phase_voltage,
                              Timeseries_objects.B_phase_voltage,
                              Timeseries_objects.C_phase_voltage,
                              Timeseries_objects.Total_power_factor,
                              Timeseries_objects.Forward_active_power,
                              Timeseries_objects.One_quadrant_wattless_power,
                              Timeseries_objects.Four_quadrant_wattless_power,
                              Timeseries_objects.year,
                              Timeseries_objects.month,
                              Timeseries_objects.day,
                              Timeseries_objects.hour,
                              Timeseries_objects.minute,
                              Timeseries_objects.second,
                              Timeseries_objects.dayofyear,
                              Timeseries_objects.week,
                              Timeseries_objects.dayofweek,
                              Timeseries_objects.time_epoch,
                              Timeseries_objects.isweekend,
                              Timeseries_objects.yesterday_label,
                              Timeseries_objects.last_week_label,
                              Timeseries_objects.max_last3_day_label,
                              Timeseries_objects.min_last3_day_label,
                              Timeseries_objects.mean_last3_day_label,
                              Timeseries_objects.median_last3_day_label,
                              Timeseries_objects.var_last3_day_label,
                              Timeseries_objects.max_last7_day_label,
                              Timeseries_objects.min_last7_day_label,
                              Timeseries_objects.mean_last7_day_label,
                              Timeseries_objects.median_last7_day_label,
                              Timeseries_objects.var_last7_day_label]]).reshape(1, -1)
        # print(lgb_array.shape)
        lgb_predict = np.expm1(lgb_model.predict(lgb_array))
        # print(lgb_predict)
        # 将lstm预测的数据整合到模型当中
        current_predict_data[keys] = 0.6 * lgb_predict[0] + 0.4 * guize_predict[0][0]
    # Get predict for 2 days
    for Timeseries_objects in Predict_timeseries_list:
        keys = str(Timeseries_objects.timeseries_time.hour) + str(':') \
               + str(Timeseries_objects.timeseries_time.minute)
        guize_predict = 0.6 * guize_A.predict(np.array([Timeseries_objects.A_phase_current]).reshape(-1, 1)) + \
                        0.4 * guize_C.predict(np.array([Timeseries_objects.C_phase_current]).reshape(-1, 1))
        # lgb预测
        lgb_array = np.array([[Timeseries_objects.Wattless_power,
                              Timeseries_objects.A_phase_current,
                              Timeseries_objects.B_phase_current,
                              Timeseries_objects.C_phase_current,
                              Timeseries_objects.A_phase_voltage,
                              Timeseries_objects.B_phase_voltage,
                              Timeseries_objects.C_phase_voltage,
                              Timeseries_objects.Total_power_factor,
                              Timeseries_objects.Forward_active_power,
                              Timeseries_objects.One_quadrant_wattless_power,
                              Timeseries_objects.Four_quadrant_wattless_power,
                              Timeseries_objects.year,
                              Timeseries_objects.month,
                              Timeseries_objects.day,
                              Timeseries_objects.hour,
                              Timeseries_objects.minute,
                              Timeseries_objects.second,
                              Timeseries_objects.dayofyear,
                              Timeseries_objects.week,
                              Timeseries_objects.dayofweek,
                              Timeseries_objects.time_epoch,
                              Timeseries_objects.isweekend,
                              Timeseries_objects.yesterday_label,
                              Timeseries_objects.last_week_label,
                              Timeseries_objects.max_last3_day_label,
                              Timeseries_objects.min_last3_day_label,
                              Timeseries_objects.mean_last3_day_label,
                              Timeseries_objects.median_last3_day_label,
                              Timeseries_objects.var_last3_day_label,
                              Timeseries_objects.max_last7_day_label,
                              Timeseries_objects.min_last7_day_label,
                              Timeseries_objects.mean_last7_day_label,
                              Timeseries_objects.median_last7_day_label,
                              Timeseries_objects.var_last7_day_label]]).reshape(1, -1)
        lgb_predict = np.expm1(lgb_model.predict(lgb_array))
        # 将lstm预测的数据整合到模型当中
        predict_data[keys] = 0.6 * lgb_predict[0] + 0.4 * guize_predict[0][0]
    """生产线信息更新"""
    # 计算当月截止目前的累计用电信息
    current_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year).\
        filter(timeseries_time__month=cdate.month).filter(timeseries_time__lte=cdate)\
        .all().aggregate(Sum('Instantaneous_active_power'))
    # 计算上个月截止到当前日的全部用电
    if cdate.month != 1:
        last_month_ele_filter_all = Timeseries.objects.filter(timeseries_time__year=cdate.year).\
            filter(timeseries_time__month=(cdate.month - 1)).filter(timeseries_time__day__lte=cdate.day)\
            .all().aggregate(Sum('Instantaneous_active_power'))
    else:
        last_month_ele_filter_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=12).all().filter(timeseries_time__day__lte=cdate.day)\
            .aggregate(Sum('Instantaneous_active_power'))
    # 计算上个月的全部用电
    if cdate.month != 1:
        last_month = cdate.month - 1
        last_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
            filter(timeseries_time__month=last_month).all().aggregate(Sum('Instantaneous_active_power'))
    else:
        last_month = 12
        last_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=12).all().aggregate(Sum('Instantaneous_active_power'))
    # 计算上上个月的全部用电
    if cdate.month not in [1, 2]:
        last2_month = cdate.month - 2
        last2_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
            filter(timeseries_time__month=(cdate.month - 2)).all().aggregate(Sum('Instantaneous_active_power'))
    elif cdate.month == 2:
        last2_month = 12
        last2_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=12).all().aggregate(Sum('Instantaneous_active_power'))
    else:
        last2_month = 11
        last2_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=11).all().aggregate(Sum('Instantaneous_active_power'))
    # 计算上上上个月的全部用电
    if cdate.month not in [1, 2, 3]:
        last3_month = cdate.month - 3
        last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
            filter(timeseries_time__month=(cdate.month - 3)).all().aggregate(Sum('Instantaneous_active_power'))
    elif cdate.month == 3:
        last3_month = 12
        last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=12).all().aggregate(Sum('Instantaneous_active_power'))
    elif cdate.month == 2:
        last3_month = 11
        last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=11).all().aggregate(Sum('Instantaneous_active_power'))
    else:
        last3_month = 10
        last3_month_ele_all = Timeseries.objects.filter(timeseries_time__year=(cdate.year - 1)). \
            filter(timeseries_time__month=10).all().aggregate(Sum('Instantaneous_active_power'))
    current_month_ele_all['Instantaneous_active_power__sum'] = \
        current_month_ele_all['Instantaneous_active_power__sum'] * 0.25
    last_month_ele_all['Instantaneous_active_power__sum'] = \
        last_month_ele_all['Instantaneous_active_power__sum'] * 0.25
    last_month_ele_filter_all['Instantaneous_active_power__sum'] = \
        last_month_ele_filter_all['Instantaneous_active_power__sum'] * 0.25
    last2_month_ele_all['Instantaneous_active_power__sum'] = \
        last2_month_ele_all['Instantaneous_active_power__sum'] * 0.25
    last3_month_ele_all['Instantaneous_active_power__sum'] = \
        last3_month_ele_all['Instantaneous_active_power__sum'] * 0.25
    huanbi = round((current_month_ele_all['Instantaneous_active_power__sum'] -
          last_month_ele_filter_all['Instantaneous_active_power__sum']) /
         last_month_ele_filter_all['Instantaneous_active_power__sum'] * 100, 2)
    if huanbi >= 0:
        huanbi_text = str('+') + str(huanbi) + str('%')
        huanbi_color_text = "#fd6265"
    else:
        huanbi_text = str(huanbi) + str('%')
        huanbi_color_text = "#4dcbb5"
    ele_fees = {}
    ele_fees[str(last3_month)] = round(last3_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
    ele_fees[str(last2_month)] = round(last2_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
    ele_fees[str(last_month)] = round(last_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
    ele_fees[str(cdate.month)] = round(current_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
    ele_fees_date = str(cdate.year) + str('.') + str(last3_month) + str('.1 - ') + str(cdate.year) + str('.') + \
                    str(cdate.month) + str('.') + str(cdate.day)
    # print('ele_fees:', ele_fees)
    # 用电检测部分数据
    current_month_ele_dict = {}
    for i in range(1, cdate.day):
        current_month_ele = Timeseries.objects.filter(timeseries_time__year=cdate.year). \
            filter(timeseries_time__month=cdate.month).filter(timeseries_time__day=i). \
            aggregate(Sum('Instantaneous_active_power'))
        current_month_ele_dict[str(i)] = current_month_ele['Instantaneous_active_power__sum']
    current_month_fees_date = str(cdate.year) + str('.') + str(cdate.month) + str('.1 - ') + str(cdate.year) + str('.') + str(cdate.month) + str('.') + str(cdate.day)
    """设置返回数据变为字典形式"""
    text_info = {}
    text_info['show_time'] = show_time
    text_info['huanbi_text'] = huanbi_text
    text_info['huanbi_color_text'] = huanbi_color_text
    text_info['current_month_ele_all'] = round(current_month_ele_all['Instantaneous_active_power__sum'], 2)
    text_info['current_month_ele_fees'] = round(current_month_ele_all['Instantaneous_active_power__sum'] * 0.725, 2)
    text_info['ele_fees_date'] = ele_fees_date
    text_info['current_month_fees_date'] = current_month_fees_date

    """设置返回数据 并且返回到js模块"""
    response = HttpResponse()
    response['Content-Type'] = "text/javascript"
    response.write('refreshData='+json.dumps({'text_info':text_info, 'current_data':current_data, 'current_predict_data':current_predict_data, \
                               'predict_data':predict_data, 'ele_fees':ele_fees, 'current_month_ele_dict':current_month_ele_dict}))
    return response
# Create your views here.
