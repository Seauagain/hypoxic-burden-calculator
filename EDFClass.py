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

class EDFClass():

    def __init__(self) -> None:
        """
        pre-processing: load edf files and read channels
        calculate: Spo2 burden
        post-processing: save results to .csv
        visulization: plot edf without/with events.
        """
        pass

    def basic_config(self,
                        patients_folder,
                        csv_name,
                        pic_folder,
                        SpO2_CSV_folder,
                        lowerSpO2Line=40
                        ):
        
        self.patients_folder = patients_folder
        self.csv_name = csv_name
        self.pic_folder  = pic_folder
        self.SpO2_CSV_folder  = SpO2_CSV_folder
        self.lowerSpO2Line = lowerSpO2Line
        import platform
        if platform.system()=='Windows':
            self.timeMark = ostime.strftime("%Y-%m-%d %H-%M-%S")
        else:
            self.timeMark = ostime.strftime("%Y-%m-%d %H:%M:%S") 
        os.makedirs(pic_folder, exist_ok=True)
        os.makedirs(SpO2_CSV_folder, exist_ok=True)



    def parse_edf(self, edf_entrance, 
                        channel_SpO2="SpO2", 
                        channel_Flow="Flow Patient-1"
                 ):
        raw = self.load_edf(edf_entrance)
        if raw:
            self.load_ok = True
        else:
            self.load_ok = False
        
        ch_names = raw.ch_names
        channel_names = [channel_SpO2, channel_Flow]

        raw_selection = raw[channel_names, :]
        self.timeline = raw_selection[1]
        self.SpO2 = raw_selection[0][0].T
        self.Flow = raw_selection[0][1].T
        self.start_record_date = raw.info['meas_date'].strftime("%Y-%m-%d %X")  # 开始记录的时间,H:M:S
        self.sfreq = raw.info['sfreq']
        ## show raw_edf info: raw.info



    def setup_logging(self):
        ###move model.py to targetPath
        log_file = os.path.join('.', self.SpO2_CSV_folder,
                                f'desaturation日志_{self.timeMark}.txt')
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


    def stand_fmt_time(self, seconds):
        """seconds to H:M:S """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)


    def time_transfer(self, time):
        """[H,M,S] 格式转换为秒数, 3600*H+60*M+S"""
        hour, min, sec = list(map(int, time))
        total_sec = 3600 * hour + 60 * min + sec
        if hour >= 12:
            return total_sec
        else:
            return total_sec + 24 * 3600  #第二天，加上24hour


    def event_start_time(self, recordTime, eventTime):
        """
        fucntion: 事件发生的时刻-->事件发生距离开始记录时刻度过的时间事件发生的时间; 
        recordTime: 开始记录的时刻; 
        eventTime: 事件发生的时刻
        """
        recordTime = recordTime.split(':')
        recordTime = self.time_transfer(recordTime)
        eventTime = self.time_transfer(eventTime)
        return eventTime - recordTime


    def event_index(self, recordTime, eventTime, eventDuration, sfreq):
        """
        事件的起始/终止index; 
        recordTime: 开始记录的时刻; 
        eventTime: 事件发生的时刻; 
        eventDuration: 事件持续时间; 
        sfreq: raw采样频率
        """
        #转为秒数方便相减
        recordTime = self.time_transfer(recordTime)
        eventTime = self.time_transfer(eventTime)
        start_idx = (eventTime - recordTime) * sfreq
        end_idx = start_idx + eventDuration * sfreq
        return int(start_idx), int(end_idx)


    def find_nearest(self, peaks, eventEndTime):
        """
        given an value, return the nearest neighbor in an array s.t. [value_left<=value<=value_right]
        return value_left,value_right
        array: shuld be time
        eventEndTime: should be a sigle value
        """
        time_select = self.timeline[peaks]
        idx_select = (np.abs(time_select - eventEndTime)).argmin()
        peak = peaks[idx_select]  # peak是对time的idx
        back_peak = peaks[idx_select - 1]
        # next_peak=peaks[np.min(idx_select+1,peaks.shape[0]-1)]
        try:
            next_peak = peaks[idx_select + 1]
        except:
            next_peak = self.timeline.shape[0] - 1
        if self.timeline[peak] <= eventEndTime:
            left_idx = self.find_end_if_flat_peaks(self.SpO2, peak, direction='right')
            right_idx = self.find_end_if_flat_peaks(self.SpO2, next_peak, direction='left')
            return left_idx, right_idx
        else:
            left_idx = self.find_end_if_flat_peaks(self.SpO2, back_peak, direction='right')
            right_idx = self.find_end_if_flat_peaks(self.SpO2, peak, direction='left')
            return left_idx, right_idx
            # return array[idx-1],array[idx]#need idx


    def find_end_if_flat_peaks(self, array, idx, direction):
        """
        find the two end points of a Flat Wave
        direction: search dicrection; 'left' or 'right'
        """
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
                raise Exception("Wrong direction, should be 'left' or 'right'")

        return idx
    
    def getSpO2BaseLine(self, endTimeIndex):
        """
        return: SpO2下降的基线，定义为事件结束前100s内SpO2的最大值
        SpO2: SpO2流
        endTime: 事件结束时间的time对应的index
        """
        end_idx = endTimeIndex
        start_idx = max(0, int(end_idx - 100 * self.sfreq))
        return np.max(self.SpO2[start_idx:end_idx + 1])


    def getSpO2Area(self, start_idx, end_idx, SpO2BaseLine):
        """氧减面积"""
        O2 = SpO2BaseLine - self.SpO2[start_idx:end_idx + 1]
        Area = np.sum(O2) / self.sfreq
        minSpO2 = np.min(self.SpO2[start_idx:end_idx + 1])
        # print("SpO2 end index-1:",SpO2[end_idx-1])
        # print("SpO2 end index:",SpO2[end_idx])
        # print("SpO2 end index+1:",SpO2[end_idx+1])
        if minSpO2 < self.lowerSpO2Line:
            return -0.00001
        else:
            return Area

    def edf_file_info(self, patient_name):
        """
        patient_name: 包含多个edf files的文件夹名
        Return [folder, Filename, edffilesNum]
        """
        subfolder = os.path.join(self.patients_folder, patient_name)
        for home, dir, files in os.walk(subfolder):
                for file in files:
                    filename = re.findall('(.*?)\[001\].edf', file)
                    if filename:
                        FilesName = filename[0]
                        break
                    else:
                        FilesName = None
                edffilesNum = 0
                # print(FilesName)
                for file in files:
                    edffiles = re.search('%s\[(.*?)\].edf' % FilesName, file)
                    if edffiles:
                        edffilesNum += 1
                # print(edffilesNum)
        # print("patient info: ",patient_name, FilesName)
        if not FilesName:
            ValueError(f"{patient_name}文件夹下没有edf")
        patientInfo = [patient_name, FilesName, edffilesNum]
        return patientInfo


    def edf_file_info_list(self):
        """
        主目录self.patients_folder下所有patients的info
        自动加载多条 [folder, FilesName, edffilesNum]
        """
        infoList = []
        patient_dir = os.path.join('.', self.patients_folder)
        for folder in os.listdir(patient_dir):
            # print("folder: ", folder)
            patientInfo = self.edf_file_info(folder)
            infoList.append(patientInfo) 
        return infoList


    # def patient_name_dict(self):
    #     """
    #     dict={patient_name: patient_order}
    #     """
    #     ## patient, FilesName, edffilesNum = infoList
    #     infoList = self.edf_files_info()
    #     name2index = {name[0]: idx for idx, name in enumerate(infoList)}
    #     return name2index


    def load_edf(self, edf_enrance):
        """ 
        加载edf，有3种输入方式:
        1. 给定edf file路径（nsrr）
        2. 给定包含多个edf files的文件夹名（Ruijin）
        3. 给定patients_folderer和patient_order（Ruijin）
        """
        if isinstance(edf_enrance, str) and edf_enrance[-4:]=='.edf':
            edf_path =  edf_enrance
            raw = mne.io.read_raw_edf(edf_path,
                                      preload=False,
                                      verbose='ERROR')
            return raw
        
        #### Ruijin format data
        # patient, FilesName, edffilesNum = '黄亚声','00000033-AN1PD2016777',9
        elif isinstance(edf_enrance, str):
            patient_name = edf_enrance
            patient, FilesName, edffilesNum = self.edf_file_info(patient_name)
        elif isinstance(edf_enrance, int):
            patient_order = edf_enrance
            infoList = self.edf_file_info_list()
            patient, FilesName, edffilesNum = infoList[patient_order]

        if edffilesNum == 0:
            return None

        rawList = []
        for i in range(1, edffilesNum + 1):
            if i <= 9:
                raw_file_path = os.path.join('.', self.patients_folder, patient,
                                            f'{FilesName}[00{i}].edf')
            else:
                raw_file_path = os.path.join('.', self.patients_folder, patient,
                                            f'{FilesName}[0{i}].edf')
            # raw_file_path = os.path.join('陈爱华/00000222-AN1PD2015224[00%s].edf' % i)
            if os.path.exists(raw_file_path):
                raw = mne.io.read_raw_edf(raw_file_path,
                                        preload=False,
                                        verbose='ERROR')
                rawList.append(raw)
        return mne.io.concatenate_raws(rawList)



    def EventNameENG(self, event_text):
        """事件，中译英"""
        result = re.search('(.*?)低通气.*?', event_text)
        if result: return 'Hypo'
        result = re.search('(.*?)呼吸暂停.*?', event_text)
        if result: return 'Apnea'
        return event_text


    def checkEventName(self, text):
        """给出text，检查事件名是否包含低通气和呼吸暂停"""
        Hypo = re.search('(.*?)低通气.*?', text)
        Apnea = re.search('(.*?)呼吸暂停.*?', text)
        if Hypo or Apnea:
            return True


    def plotChannels(self, SpO2BaseLine, eventName, 
                    patient_name, start_time, end_time, 
                    left_idx, right_idx, show_event):
        #time, Flow, SpO2, sfreq,
        """
        绘制带有事件信息的channel signal
        """
        start_idx = np.where(start_time == self.timeline)[0]
        idx = min(left_idx, start_idx)
        idx1 = int(max(0, idx // self.sfreq - 10) * self.sfreq)
        idx2 = int((right_idx // self.sfreq + 10) * self.sfreq)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(self.timeline[idx1:idx2], self.Flow[idx1:idx2], color='blue')
      
        ax1.set_ylabel('Flow', color='blue', fontweight='bold')
        #####
        ax2.plot(self.timeline[idx1:idx2], self.SpO2[idx1:idx2], color='red')
        
        ax2.set_xlabel('time(s)', fontweight='bold')
        ax2.set_ylabel('SpO2(%)', color='red', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        if show_event:# 是否展示事件相关的文字信息

            ax1.fill_between(x=[start_time, end_time],
                        y1=np.min(self.Flow[idx1:idx2]),
                        y2=np.max(self.Flow[idx1:idx2]),
                        color='gray',
                        alpha=0.4)
            # ax1.vlines
            ax1.text(x=start_time * 1 + 0.2 * (end_time - start_time),
                    y=np.max(self.Flow[idx1:idx2]) * 0.9,
                    s=eventName,
                    fontweight='bold')
            ax2.fill_between(self.timeline[left_idx:right_idx],
                        y1=self.SpO2[left_idx:right_idx],
                        y2=SpO2BaseLine,
                        color='gray',
                        alpha=0.8)
            ax2.plot(self.timeline[left_idx],  self.SpO2[left_idx], '>', markersize=5, color='k')
            ax2.plot(self.timeline[right_idx], self.SpO2[right_idx], '<', markersize=5, color='k')

        
        plt.subplots_adjust(wspace=0, hspace=0)
        pic_dir = os.path.join(
            '.', self.pic_folder,
            f'{patient_name}startTime={start_time}_endTime={end_time}.png')
        plt.savefig(pic_dir, dpi=200)
        print(f'event pic saved in {pic_dir}')
        plt.close()


    def plotPatientSpO2(self, patient_name, start2end_time_list, show_event=True):
        """
        可视化SpO2和Flow的信号图
        patient_name: 三种输入方式edf_path/patient_name/patient_order
        show_event: 绘图时是否展示与事件相关的信息，默认True
        """
        

        # patient_names_dict = self.patient_name_list()
        # patient_idx = patient_names_dict[patient_name]
        # raw, patient = self.load_edf(patient_idx)
        # startDate = raw.info['meas_date'].strftime("%Y-%m-%d %X")  # 开始记录的时间,H:M:S
        # sfreq = raw.info['sfreq']
        # channel_names = ['SpO2', 'Flow Patient-1']
        
        self.parse_edf(patient_name)
        start_time, end_time = start2end_time_list
        start_time = int(start_time)
        end_time = int(end_time)

        # start_sec = (max(0, start_time - 100)) * sfreq  # timeForPeaks
        # stop_sec = (end_time + 100) * sfreq  #timeForPeaks
        # raw_selection = raw[channel_names, start_sec:stop_sec]  #start_sec:stop_sec
        # raw_selection = raw[channel_names, :]
        # time = raw_selection[1]
        # SpO2 = raw_selection[0][0].T
        # Flow = raw_selection[0][1].T

        peaks, _ = find_peaks(self.SpO2, height=0)  #peaks指标
        left_idx, right_idx = self.find_nearest(peaks, end_time)
        SpO2BaseLine = self.getSpO2BaseLine(right_idx)

        #Flow阴影区的起始时间(int): start_time, end_time
        #寻找peaks的时间timeForPeaks(np.arr): start_time, end_time
        #坐标轴绘图的时间timeForPlots(np.arr): start_time-10
        if show_event: #读取csv，获得event信息
            csv_dir = os.path.join('.', self.patients_folder, patient_name, self.csv_name)
            f = open(csv_dir, encoding="utf-8")
            df = pd.read_csv(f)

            ####
            # startDate = raw.info['meas_date'].strftime(
            #     "%Y-%m-%d %X")  # 开始记录的时间,H:M:S，win10路径不能出现冒号
            date0, recordTime = self.start_record_date.split()
            df['开始时间'] = df['时间'].str.split(':')
            df['开始时间'] = pd.DataFrame(
                self.event_start_time(recordTime, x) for x in df['开始时间'])
            Start = df['开始时间'].values
            row = np.where(abs(start_time - Start) < 1)[0]
            eventName = self.EventNameENG(str(df.loc[row]['类型']))
        # print("eventname: ",eventName)
        self.plotChannels(SpO2BaseLine, eventName,
                    patient_name, start_time, end_time, 
                    left_idx, right_idx,show_event)




    def computeSpO2(self, mode='all'):
        ##计算SpO2，默认计算所有病人
        ##可以指定计算某个病人
        #初始化log
        self.setup_logging()
        #更新病人名单
        infoList = self.edf_file_info_list()
        patient_names_list = [info[0] for info in infoList] # all patients
        # patient_names_dict = self.patient_name_list()
        if mode == 'all':
            patient_names_select =  patient_names_list
            # patient_names_list = list(self.patient_name_list().values())
        else:
            patient_names_select = [mode]
        toal_num = len(patient_names_list)
        logging.info(f'需要计算的总人数={toal_num:^3}\n')

        for patient_name in patient_names_select:
            patient_order = patient_names_list.index(patient_name)
            self.parse_edf(patient_name)
            ###TODO：判断异常情况
            if not self.load_ok:
                logging.info(f'index={patient_order:^3} patient {patient_name:^6}没有EDF文件\n')
                continue
            csv_dir = os.path.join('.', self.patients_folder, patient_name, self.csv_name)
            try:
                f = open(csv_dir, encoding="utf-8")
                df = pd.read_csv(f)
            except:
                logging.info(
                    f"index={patient_order:^3} patient {patient_name:^6}没有<{self.csv_name}>文件\n")
                continue
            date0, recordTime = self.start_record_date.split()
            df['开始时间'] = df['时间'].str.split(':')
            df['开始时间'] = pd.DataFrame(
                self.event_start_time(recordTime, x) for x in df['开始时间'])
            df['结束时间'] = df['开始时间'] + df['持续时间']

            SpO2Area = []

            # raw_selection = raw[channel_names, :]
            # time = raw_selection[1]
            # SpO2 = raw_selection[0][0].T
            # Flow = raw_selection[0][1].T

            for row in range(df.shape[0]):
                eventName = df.loc[row]['类型']
                if self.checkEventName(eventName):
                    eventEndTime = df.loc[row]['结束时间']
                    # if eventEndTime>time[-1] or eventEndTime<time[0]:
                    #       SpO2Area.append(-0.01)
                    # else:
                    peaks, _ = find_peaks(self.SpO2, height=0)  #peaks指标
                    left_idx, right_idx = self.find_nearest(peaks,
                                                    eventEndTime)  #End是时间
                    SpO2BaseLine = self.getSpO2BaseLine(right_idx)
                    Area = self.getSpO2Area(left_idx, right_idx, SpO2BaseLine)
                else:
                    Area = 0
                SpO2Area.append(Area)

            df['氧减面积'] = SpO2Area
            df['氧减面积总和(%·s)'] = df['氧减面积'].sum()

            logging.info(f'index={patient_order:^3} patient {patient_name:^6}')
            logging.info(f'record time: {recordTime}')
            logging.info('氧减面积求和: {:.3f}(%·s)'.format(df['氧减面积'].sum()))
            logging.info('睡眠总时长: %s\n' % self.stand_fmt_time(self.timeline[-1]))

            df.sort_values(by='开始时间', inplace=True)
            #分别保存csv在对应patient文件夹和单独的SpO2文件夹
            csv_to_patient = os.path.join(self.patients_folder, patient_name,
                                        f'{patient_name}desaturation.csv')
            df.to_csv(csv_to_patient, encoding='utf_8_sig')
            csv_to_spo2 = os.path.join(self.SpO2_CSV_folder, f'{patient_name}desaturation.csv')
            df.to_csv(csv_to_spo2, encoding='utf_8_sig')


            # my_annot = mne.Annotations(
            #     onset=df['开始时间'].tolist(),  # in seconds
            #     duration=df['持续时间'].tolist(),  # in seconds, too
            #     description=df['类型'].tolist())
            # raw.set_annotations(my_annot)

            # # Start=raw.annotations.onset
            # # Duration=raw.annotations.duration
            # # End=Start+Duration


            

    def overallCSV(self):
        ##汇总所有patient的计算结果保存到overall_timeMark.csv
        subfolder = os.path.join('.', self.SpO2_CSV_folder)
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
        resDir = os.path.join('.', self.SpO2_CSV_folder,
                            f'[0]汇总{len(resultList)}人计算结果.csv')
        newdf.to_csv(resDir, encoding="utf_8_sig", index=True)
        print(f'final overall-result saved in {resDir}')


