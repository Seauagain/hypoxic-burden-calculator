from math import hypot
import mne
import matplotlib.pyplot as plt
from mne.transforms import rotation
import numpy as np
import pandas as pd
import os
import matplotlib
import re

matplotlib.use('AGG')


def patientFiles():
    '''自动加载'''
    infoList = []
    for folder in os.listdir('./wangyi_data'):
        # result = re.search('.npy', file)
        # print(folder)
        for home, dir, files in os.walk('./wangyi_data/%s' % folder):
            for file in files:
                filename = re.findall('(.*?).rml', file)
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


def combineEDF():
    infoList = patientFiles()
    patient, FilesName, edffilesNum = infoList[1]
    rawList = []
    for i in range(1, edffilesNum + 1):
        # text = '/wangyi_data//00000111-AN1PD2016546'
        # raw_file_path=os.path.join('武茵/00002750-A5BS16542[00%s].edf'%i)
        if i <= 9:
            raw_file_path = os.path.join('./wangyi_data/%s/%s[00%i].edf' %
                                         (patient, FilesName, i))
        else:
            raw_file_path = os.path.join('./wangyi_data/%s/%s[0%i].edf' %
                                         (patient, FilesName, i))
        # raw_file_path = os.path.join('陈爱华/00000222-AN1PD2015224[00%s].edf' % i)
        raw = mne.io.read_raw_edf(raw_file_path,
                                  preload=False,
                                  verbose='ERROR')
        rawList.append(raw)
    return mne.io.concatenate_raws(rawList), patient


# raw_file_path = os.path.join('薛祖平/00002689-A5BS15717[002].edf')
# raw1=mne.io.read_raw_edf(raw_file_path,preload=False,verbose=False)


def plotChannels(time, Flow, SpO2, FlowBaseLine):
    # with plt.style.context(['science']):
    print('first:', raw.first_samp)
    events = mne.find_events(raw,
                             stim_channel=['Flow Patient'],
                             shortest_event=1)  #Flow Patient
    event_times = (events[:, 0] - raw.first_samp).astype(float)
    event_times /= sfreq
    event_nums = events[:, 2]
    if True:
        fig, ax1 = plt.subplots()
        ax1.plot(time, SpO2, color='red')
        # print('ID:', event_nums)
        # print('event_time', event_times)

        ax1.vlines(event_times[:1000],
                   0,
                   100,
                   linestyles='dashed',
                   colors='red')
        # ax1.plot(event_times, event_nums)
        ax1.set_xlabel('time(s)', fontweight='bold')
        ax1.set_ylabel('SpO2', color='red', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(time, Flow, color='blue')
        ax2.set_ylabel('Flow', color='blue', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.plot(time, 0 * time + 0.1 * FlowBaseLine, color='black')
        ax2.plot(time, 0 * time - 0.1 * FlowBaseLine, color='black')
        plt.savefig('test.png', dpi=200)
        plt.close()


def getHypoPiece(Flow, FlowBaseLine):
    '''
    return: 呼吸暂停（Hypopnea）的起始与停止的index
    '''
    Flow = np.abs(Flow.copy())
    Flow[Flow > 0.1 * FlowBaseLine] = 0
    l = Flow.tolist()
    df = pd.DataFrame({'A': l})
    df['block'] = (df['A'] == 0).astype(int).cumsum()  # 对等于0的进行累加计算。
    df = df.reset_index()
    df = df[df.A != 0]  # 删除掉为0的元素
    list = df.groupby(['block'])['index'].apply(np.array).tolist()  #连续的片段
    Index = []
    for arr in list:
        start_end_index = [arr[0], arr[-1]]
        Index.append(start_end_index)
    return Index


def getApneaPiece(Flow, FlowBaseLine):
    '''
    return: 低通气（Apnea）的起始与停止的index
    '''
    Flow = np.abs(Flow.copy())
    Flow[Flow <= 0.1 * FlowBaseLine] = 0
    Flow[Flow > 0.7 * FlowBaseLine] = 0
    l = Flow.tolist()
    df = pd.DataFrame({'A': l})
    df['block'] = (df['A'] == 0).astype(int).cumsum()  # 对等于0的进行累加计算。
    df = df.reset_index()
    df = df[df.A != 0]  # 删除掉为0的元素
    list = df.groupby(['block'])['index'].apply(np.array).tolist()  #连续的片段
    Index = []
    for arr in list:
        start_end_index = [arr[0], arr[-1]]
        Index.append(start_end_index)
    return Index


def getSpO2BaseLine(SpO2, endTimeIndex, sfreq):
    '''
    return: SpO2下降的基线，定义为事件结束前100s内SpO2的最大值
    SpO2: SpO2流
    endTime: 事件结束时间的time对应的index
    '''
    end_idx = endTimeIndex
    start_idx = max(0, int(end_idx - 100 * sfreq))
    return np.max(SpO2[start_idx:end_idx + 1])


def getFlowBaseLine():
    '''return: Flow下降基线'''
    pass


def getSpO2HypoArea(SpO2, sfreq, start_idx, end_idx, SpO2BaseLine):
    '''呼吸暂停的氧减面积'''
    O2 = SpO2BaseLine - SpO2[start_idx:end_idx]
    Area = np.sum(O2) / sfreq
    return Area


def getSpO2ApneaArea(SpO2, sfreq, start_idx, end_idx, SpO2BaseLine):
    '''低通气的氧减面积'''
    O2 = SpO2BaseLine - SpO2[start_idx:end_idx]
    Area = np.sum(O2) / sfreq
    return Area
    pass


raw, patient = combineEDF()

channel_names = ['SpO2', 'Flow Patient']
sfreq = raw.info['sfreq']
start_sec = 20 * sfreq
stop_sec = 30 * sfreq
raw_selection = raw[channel_names, start_sec:stop_sec]  #start_sec:stop_sec

# raw_selection = raw[channel_names, :]
time = raw_selection[1]
SpO2 = raw_selection[0][0].T
Flow = raw_selection[0][1].T
# Flow = np.abs(Flow)

######
FlowBaseLine = np.max(np.abs(Flow))
EventDuration = 10  # 事件持续时间
plotChannels(time, Flow, SpO2, FlowBaseLine)

# for event in raw.annotations:
#     print("event", event)
# print('ann:', raw.annotations)
# IndexHpyo = getHypoPiece(Flow, FlowBaseLine)
# IndexApnea = getApneaPiece(Flow, FlowBaseLine)
# HypoNum = 0  # 记录呼吸暂停
# ApneaNum = 0  # 记录低通气
# HypoArea = []
# ApneaArea = []
# for idx in IndexHpyo:
#     start_idx = idx[0]
#     end_idx = idx[1]
#     if time[end_idx] - time[start_idx] > EventDuration:
#         # print('呼吸暂停 start={:.3f}s, duration={:.3f}s'.format(
#         #     time[start_idx], time[end_idx] - time[start_idx]))
#         HypoNum += 1
#         SpO2BaseLine = getSpO2BaseLine(SpO2, end_idx, sfreq)
#         Area = getSpO2HypoArea(SpO2, sfreq, start_idx, end_idx, SpO2BaseLine)
#         HypoArea.append(Area)
#         # print('SpO2BaseLine:{:.3f} 呼吸暂停 Area:{:.3f}'.format(
#         #     SpO2BaseLine, Area))

# for idx in IndexApnea:
#     start_idx = idx[0]
#     end_idx = idx[1]
#     if time[end_idx] - time[start_idx] > EventDuration:
#         # print('低通气 start={:.3f}s, duration={:.3f}s'.format(
#         #     time[start_idx], time[end_idx] - time[start_idx]))
#         ApneaNum += 1
#         SpO2BaseLine = getSpO2BaseLine(SpO2, end_idx, sfreq)
#         Area = getSpO2ApneaArea(SpO2, sfreq, start_idx, end_idx, SpO2BaseLine)
#         ApneaArea.append(Area)
#         # print('SpO2BaseLine:{:.3f} 低通气 Area:{:.3f}'.format(SpO2BaseLine, Area))

# print(patient)
# print('呼吸暂停', HypoNum)
# print('低通气', ApneaNum)
# Area = HypoArea + ApneaArea
# print(np.sum(Area) / 100)

# events = mne.find_events(raw, stim_channel=['Flow Patient'])
# event_times = (events[:, 0] - raw.first_samp).astype(float)
# event_times /= sfreq
# event_nums = events[:, 2]

# fig = raw.plot(events=events, event_color='red', duration=200, scalings='auto')
# fig.savefig('events', dpi=200)
#
#         SpO2BaseLine=getSpO2BaseLine(SpO2,end_idx,sfreq)
#         Area=getSpO2HypoArea(SpO2,sfreq,start_idx,end_idx,SpO2BaseLine)
#         print('SpO2BaseLine:{:.3f} Area:{:.3f}'.format(SpO2BaseLine,Area))
