import mne
import os,re,sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
import time as ostime


matplotlib.use('AGG')
# matplotlib.rcParams.update({"figure.facecolor": "white"})


def setup_logging():
    ###move model.py to targetPath
    log_file = os.path.join('.', SpO2_CSV_fold,
                            f'desaturation日志_{timeMark}.txt')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    # create streamhandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create filehandler
    sh = logging.FileHandler(str(log_file))
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def standFmtTime(seconds):
    '''seconds to H:M:S '''
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def timeTransfer(time):
    '''[H,M,S] 格式转换为秒数, 3600*H+60*M+S'''
    hour, min, sec = list(map(int, time))
    total_sec = 3600 * hour + 60 * min + sec
    if hour >= 12:
        return total_sec
    else:
        return total_sec + 24 * 3600  #第二天，加上24hour


def eventStartTime(recordTime, eventTime):
    '''
    fucntion: 事件发生的时刻-->事件发生距离开始记录时刻度过的时间
    事件发生的时间; recordTime: 开始记录的时刻; eventTime: 事件发生的时刻
    '''
    recordTime = recordTime.split(':')
    recordTime = timeTransfer(recordTime)
    eventTime = timeTransfer(eventTime)
    return eventTime - recordTime


def eventIndex(recordTime, eventTime, eventDuration, sfreq):
    '''事件的起始/终止index; recordTime: 开始记录的时刻; eventTime: 事件发生的时刻; eventDuration: 事件持续时间; sfreq: raw采样频率'''
    #转为秒数方便相减
    recordTime = timeTransfer(recordTime)
    eventTime = timeTransfer(eventTime)
    start_idx = (eventTime - recordTime) * sfreq
    end_idx = start_idx + eventDuration * sfreq
    return int(start_idx), int(end_idx)


def findNearest(time, SpO2, peaks, eventEndTime):
    '''
    given an value, return the nearest neighbor in an array s.t. [value_left<=value<=value_right]
    return value_left,value_right
    array: shuld be time
    eventEndTime: should be a sigle value
    '''
    time_select = time[peaks]
    idx_select = (np.abs(time_select - eventEndTime)).argmin()
    peak = peaks[idx_select]  # peak是对time的idx
    back_peak = peaks[idx_select - 1]
    # next_peak=peaks[np.min(idx_select+1,peaks.shape[0]-1)]
    try:
        next_peak = peaks[idx_select + 1]
    except:
        next_peak = time.shape[0] - 1
    if time[peak] <= eventEndTime:
        left_idx = findEndifFlatPeaks(SpO2, peak, direction='right')
        right_idx = findEndifFlatPeaks(SpO2, next_peak, direction='left')
        return left_idx, right_idx
    else:
        left_idx = findEndifFlatPeaks(SpO2, back_peak, direction='right')
        right_idx = findEndifFlatPeaks(SpO2, peak, direction='left')
        return left_idx, right_idx
        # return array[idx-1],array[idx]#need idx


def findEndifFlatPeaks(array, idx, direction):
    '''
    find the two end points of a Flat Wave
    direction: search dicrection; 'left' or 'right'
    '''
    idx, peaksValue = idx, array[idx]
    while True:
        if direction == 'right':
            nextValue = array[idx + 1]
            if nextValue == peaksValue:
                idx = idx + 1
            else:
                break
        elif direction == 'left':
            nextValue = array[idx - 1]
            if nextValue == peaksValue:
                idx = idx - 1
            else:
                break
        else:
            raise Exception("Wrong direction, should be'left' or 'right'")

    return idx


def patientFiles():
    '''自动加载'''
    infoList = []
    
    patient_dir = os.path.join('.', patient_fold)
    for folder in os.listdir(patient_dir):
        # result = re.search('.npy', file)
        # print(folder)
        subfolder = os.path.join(patient_fold, folder)
        for home, dir, files in os.walk(subfolder):
            for file in files:
                filename = re.findall('(.*?)\[001\].edf', file)
                if filename:
                    FilesName = filename[0]
                    break
            edffilesNum = 0
            # print(FilesName)
            for file in files:
                edffiles = re.search('%s\[(.*?)\].edf' % FilesName, file)
                if edffiles:
                    edffilesNum += 1
            # print(edffilesNum)
            patientInfo = [folder, FilesName, edffilesNum]
            infoList.append(patientInfo)
    return infoList


def patientNameList():
    info = patientFiles()
    name2index = {name[0]: idx for idx, name in enumerate(info)}
    return name2index


def combineEDF(patient_order):
    # patient, FilesName, edffilesNum = '黄亚声','00000033-AN1PD2016777',9
    infoList = patientFiles()
    # print('len of infoList', len(infoList))
    patient, FilesName, edffilesNum = infoList[patient_order]
    # print(patient)
    if edffilesNum == 0:
        return None, patient

    rawList = []
    for i in range(1, edffilesNum + 1):
        if i <= 9:
            raw_file_path = os.path.join('.', patient_fold, patient,
                                         f'{FilesName}[00{i}].edf')
        else:
            raw_file_path = os.path.join('.', patient_fold, patient,
                                         f'{FilesName}[0{i}].edf')
        # raw_file_path = os.path.join('陈爱华/00000222-AN1PD2015224[00%s].edf' % i)
        if os.path.exists(raw_file_path):
            raw = mne.io.read_raw_edf(raw_file_path,
                                      preload=False,
                                      verbose='ERROR')
            rawList.append(raw)
    return mne.io.concatenate_raws(rawList), patient


def getSpO2BaseLine(SpO2, endTimeIndex, sfreq):
    '''
    return: SpO2下降的基线，定义为事件结束前100s内SpO2的最大值
    SpO2: SpO2流
    endTime: 事件结束时间的time对应的index
    '''
    end_idx = endTimeIndex
    start_idx = max(0, int(end_idx - 100 * sfreq))
    return np.max(SpO2[start_idx:end_idx + 1])


def getSpO2Area(SpO2, sfreq, start_idx, end_idx, SpO2BaseLine):
    '''氧减面积'''
    O2 = SpO2BaseLine - SpO2[start_idx:end_idx + 1]
    Area = np.sum(O2) / sfreq
    minSpO2 = np.min(SpO2[start_idx:end_idx + 1])
    # print("SpO2 end index-1:",SpO2[end_idx-1])
    # print("SpO2 end index:",SpO2[end_idx])
    # print("SpO2 end index+1:",SpO2[end_idx+1])
    if minSpO2 < lowerSpO2Line:
        return -0.00001
    else:
        return Area


def EventNameENG(strEvent):
    '''事件，中译英'''
    result = re.search('(.*?)低通气.*?', strEvent)
    if result: return 'Hypo'
    result = re.search('(.*?)呼吸暂停.*?', strEvent)
    if result: return 'Apnea'
    return strEvent


def checkEventName(str):
    '''给出str，检查事件名是否包含低通气和呼吸暂停'''
    Hypo = re.search('(.*?)低通气.*?', str)
    Apnea = re.search('(.*?)呼吸暂停.*?', str)
    if Hypo or Apnea:
        return True


def plotChannels(SpO2BaseLine, eventName, time, Flow, SpO2, sfreq,
                 patient_name, start_time, end_time, left_idx, right_idx):

    start_idx = np.where(start_time == time)[0]
    idx = min(left_idx, start_idx)
    idx1 = int(max(0, idx // sfreq - 10) * sfreq)
    idx2 = int((right_idx // sfreq + 10) * sfreq)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(time[idx1:idx2], Flow[idx1:idx2], color='blue')
    ax1.fill_between(x=[start_time, end_time],
                     y1=np.min(Flow[idx1:idx2]),
                     y2=np.max(Flow[idx1:idx2]),
                     color='gray',
                     alpha=0.4)
    # ax1.vlines
    ax1.text(x=start_time * 1 + 0.2 * (end_time - start_time),
             y=np.max(Flow[idx1:idx2]) * 0.9,
             s=eventName,
             fontweight='bold')
    ax1.set_ylabel('Flow', color='blue', fontweight='bold')
    #####
    ax2.plot(time[idx1:idx2], SpO2[idx1:idx2], color='red')
    ax2.fill_between(time[left_idx:right_idx],
                     y1=SpO2[left_idx:right_idx],
                     y2=SpO2BaseLine,
                     color='gray',
                     alpha=0.8)

    ax2.plot(time[left_idx], SpO2[left_idx], '>', markersize=5, color='k')
    ax2.plot(time[right_idx], SpO2[right_idx], '<', markersize=5, color='k')

    ax2.set_xlabel('time(s)', fontweight='bold')
    ax2.set_ylabel('SpO2(%)', color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.subplots_adjust(wspace=0, hspace=0)
    pic_dir = os.path.join(
        '.', pic_fold,
        f'{patient_name}startTime={start_time}_endTime={end_time}.png')
    plt.savefig(pic_dir, dpi=200)
    print(f'event pic saved in {pic_dir}')
    plt.close()


def plotPatientSpO2(patient_name, start2end_timelist):
    ##可视化病人的事件
    patient_names_dict = patientNameList()
    patient_idx = patient_names_dict[patient_name]
    raw, patient = combineEDF(patient_idx)
    startDate = raw.info['meas_date'].strftime("%Y-%m-%d %X")  # 开始记录的时间,H:M:S
    sfreq = raw.info['sfreq']
    channel_names = ['SpO2', 'Flow Patient-1']

    start_time, end_time = start2end_timelist
    start_time = int(start_time)
    end_time = int(end_time)

    # start_sec = (max(0, start_time - 100)) * sfreq  # timeForPeaks
    # stop_sec = (end_time + 100) * sfreq  #timeForPeaks
    # raw_selection = raw[channel_names, start_sec:stop_sec]  #start_sec:stop_sec
    raw_selection = raw[channel_names, :]
    time = raw_selection[1]
    SpO2 = raw_selection[0][0].T
    Flow = raw_selection[0][1].T

    peaks, _ = find_peaks(SpO2, height=0)  #peaks指标
    left_idx, right_idx = findNearest(time, SpO2, peaks, end_time)
    SpO2BaseLine = getSpO2BaseLine(SpO2, right_idx, sfreq)

    #Flow阴影区的起始时间(int): start_time, end_time
    #寻找peaks的时间timeForPeaks(np.arr): start_time, end_time
    #坐标轴绘图的时间timeForPlots(np.arr): start_time-10

    csv_dir = os.path.join('.', patient_fold, patient, csv_name)
    f = open(csv_dir, encoding="utf-8")
    df = pd.read_csv(f)

    ####
    startDate = raw.info['meas_date'].strftime(
        "%Y-%m-%d %X")  # 开始记录的时间,H:M:S，win10路径不能出现冒号
    date0, recordTime = startDate.split()
    df['开始时间'] = df['时间'].str.split(':')
    df['开始时间'] = pd.DataFrame(
        eventStartTime(recordTime, x) for x in df['开始时间'])
    Start = df['开始时间'].values
    row = np.where(abs(start_time - Start) < 1)[0]
    eventName = EventNameENG(str(df.loc[row]['类型']))
    ####
    plotChannels(SpO2BaseLine, eventName, time, Flow, SpO2, sfreq,
                 patient_name, start_time, end_time, left_idx, right_idx)


def computeSpO2(mode='all'):
    ##计算SpO2，默认计算所有病人
    ##可以指定计算某个病人
    #初始化log
    setup_logging()
    #更新病人名单
    patient_names_dict = patientNameList()
    if mode == 'all':
        patient_names_list = list(patientNameList().values())
    else:
        patient_names_list = [patient_names_dict[mode]]
    toal_num = len(patient_names_list)
    logging.info(f'需要计算的总人数={toal_num:^3}\n')

    for i in patient_names_list:
        patient_order = i
        raw, patient = combineEDF(patient_order)
        ###TODO：判断异常情况
        if not raw:
            logging.info(f'index={i:^3} patient {patient:^6}没有EDF文件\n')
            continue
        csv_dir = os.path.join('.', patient_fold, patient, csv_name)
        try:
            f = open(csv_dir, encoding="utf-8")
        except:
            logging.info(
                f"index={i:^3} patient {patient:^6}没有<{csv_name}>文件\n")
            continue

        sfreq = raw.info['sfreq']
        channel_names = ['SpO2', 'Flow Patient-1']

        df = pd.read_csv(f)
        startDate = raw.info['meas_date'].strftime(
            "%Y-%m-%d %X")  # 开始记录的时间,H:M:S
        date0, recordTime = startDate.split()
        df['开始时间'] = df['时间'].str.split(':')
        df['开始时间'] = pd.DataFrame(
            eventStartTime(recordTime, x) for x in df['开始时间'])
        df['结束时间'] = df['开始时间'] + df['持续时间']
        my_annot = mne.Annotations(
            onset=df['开始时间'].tolist(),  # in seconds
            duration=df['持续时间'].tolist(),  # in seconds, too
            description=df['类型'].tolist())
        raw.set_annotations(my_annot)
        # Start=raw.annotations.onset
        # Duration=raw.annotations.duration
        # End=Start+Duration

        SpO2Area = []
        raw_selection = raw[channel_names, :]
        time = raw_selection[1]
        SpO2 = raw_selection[0][0].T
        Flow = raw_selection[0][1].T

        for row in range(df.shape[0]):
            eventName = df.loc[row]['类型']
            if checkEventName(eventName):
                eventEndTime = df.loc[row]['结束时间']
                # if eventEndTime>time[-1] or eventEndTime<time[0]:
                #       SpO2Area.append(-0.01)
                # else:
                peaks, _ = find_peaks(SpO2, height=0)  #peaks指标
                left_idx, right_idx = findNearest(time, SpO2, peaks,
                                                  eventEndTime)  #End是时间
                SpO2BaseLine = getSpO2BaseLine(SpO2, right_idx, sfreq)
                Area = getSpO2Area(SpO2, sfreq, left_idx, right_idx,
                                   SpO2BaseLine)
            else:
                Area = 0
            SpO2Area.append(Area)

        df['氧减面积'] = SpO2Area
        df['氧减面积总和(%·s)'] = df['氧减面积'].sum()

        logging.info(f'index={i:^3} patient {patient:^6}')
        logging.info(f'record time: {recordTime}')
        logging.info('氧减面积求和: {:.3f}(%·s)'.format(df['氧减面积'].sum()))
        logging.info('睡眠总时长: %s\n' % standFmtTime(time[-1]))

        df.sort_values(by='开始时间', inplace=True)
        #分别保存csv在对应patient文件夹和单独的SpO2文件夹
        csv_to_patient = os.path.join(patient_fold, patient,
                                      f'{patient}desaturation.csv')
        df.to_csv(csv_to_patient, encoding='utf_8_sig')
        csv_to_spo2 = os.path.join(SpO2_CSV_fold, f'{patient}desaturation.csv')
        df.to_csv(csv_to_spo2, encoding='utf_8_sig')


def overallCSV():
    ##汇总所有patient的计算结果保存到overall_timeMark.csv
    subfolder = os.path.join('.', SpO2_CSV_fold)
    resultList = []
    for home, dir, files in os.walk(subfolder):
        for file in files:
            # print(home, dir, file)
            filename = re.findall('(.*?)desaturation.csv', file)
            if filename:
                tmpRes = {}
                patient_name = filename[0]
                file_dir = os.path.join(subfolder, file)
                df = pd.read_csv(file_dir, encoding="utf-8")
                tmpRes['姓名'] = patient_name
                tmpRes['氧减面积总和(%·s)'] = round(df.loc[0]['氧减面积总和(%·s)'], 2)
                resultList.append(tmpRes)
            else:
                continue
    if not resultList:
        print('没有找到desaturation.csv文件')
        return

    newdf = pd.DataFrame(resultList)
    resDir = os.path.join('.', SpO2_CSV_fold,
                          f'[0]汇总{len(resultList)}人计算结果.csv')
    newdf.to_csv(resDir, encoding="utf_8_sig", index=True)
    print(f'final overall-result saved in {resDir}')


patient_fold = '问卷PSG_and_事件'
csv_name ='全部.csv'
pic_fold = 'SpO2Pic'