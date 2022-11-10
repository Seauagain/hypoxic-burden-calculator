from EDF_Events import *

if __name__ == '__main__':
    ##TODO: spo2计算模式，默认计算所有病人。
    ##patient_name: 'all':计算所有病人；'武茵': 计算某个指定的病人
    ##patient_name = '周胜伟'
    patient_name = 'all'
    #computeSpO2(patient_name)  #计算SpO2，不需要重复计算SpO2就把这句注释掉
    #overallCSV()
    ##TODO: 可视化病人的事件，图片保存在SpO2Pic
    ##For instance: 打开周波.desaturation.csv，最后几列分别是事件对应的开始时间和结束时间，
    ##patient_time：病人名字
    ##start2end_list：事件的起始时间
    patient_name = '薛祖平'
    start2end_timelist = [4349, 4361]
    plotPatientSpO2(patient_name, start2end_timelist)
