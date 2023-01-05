from EDFClass import EDFClass
import matplotlib.pyplot as plt
import os


def main():

    EDF = EDFClass()
    patients_folder = '问卷PSG_and_事件'
    csv_name = '全部.csv'
    pic_folder = 'SpO2Pic'
    SpO2_CSV_folder = 'SpO2Area'
    lowerSpO2Line = 40  # 默认40
    EDF.basic_config(patients_folder, csv_name, pic_folder, SpO2_CSV_folder,
                     lowerSpO2Line)

    xml_files_home = os.path.join('..', 'polysomnography',
                                  'annotations-events-nsrr', 'shhs1')
    edf_files_home = os.path.join('..', 'polysomnography', 'edfs', 'shhs1')
    EDF.nsrr_config(xml_files_home, edf_files_home)

    ##TODO: spo2计算模式，默认计算所有病人。
    ##patient_name: 'all':计算所有病人；'武茵': 计算某个指定的病人

    # data_type = 'ruijin'  #数据格式 ruijin
    data_type = 'nsrr'
    # patient_name = '白川'
    patient_name = 'all'
    EDF.computeSpO2(patient_name, data_type)  #计算SpO2，不需要重复计算SpO2就把这句注释掉
    # EDF.overallCSV()

    ##TODO: 可视化病人的事件，图片保存在SpO2Pic
    ##For instance: 打开周波.desaturation.csv，最后几列分别是事件对应的开始时间和结束时间，
    ##patient_time：病人名字
    ##start2end_time_list：事件的起始时间
    # patient_name = '周波'
    edf_path = "../polysomnography/edfs/shhs1/shhs1-200002.edf"
    start2end_time_list = [13, 30]
    # EDF.plotPatientSpO2(patient_name, start2end_time_list, show_event=1)
    # EDF.plotPatientSpO2(edf_path, start2end_time_list, data_type, show_event=1)


if __name__ == '__main__':
    # with plt.style.context(['science']):
    main()
