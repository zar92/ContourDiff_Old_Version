
import argparse
import pandas as pd
import DataProcessing
import GenerateImages
import matplotlib.pyplot as plt
import numpy as np


def main(list_of_files,column):
    file_list = list()
    for i in range(len(list_of_files)):
        data = pd.read_csv('Dataset/'+list_of_files[i])[['longitude','latitude',column]]
        data.to_csv('Dataset/'+'data'+str(i)+'.csv')
        file_list.append('Dataset/'+'data'+str(i)+'.csv')
    data = DataProcessing.importData(file_list[0])
    data['levels'] = (data[column] - data[column].min())/(data[column].max() - data[column].min())
    levels = [0.25,0.5,0.75]
    cntr_set=plt.contour(np.array(data['levels']).reshape(699,639),levels,colors=['g','r','y'])
    cntr_data = DataProcessing.modelTheGraph(cntr_set)
    weighted_graph = DataProcessing.createWeightedGraph(cntr_data,file_list,column)
    filtered_graph = DataProcessing.filterBasedOnGrid(10,weighted_graph)
    GenerateImages.generateImages(filtered_graph,data,column)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contourdiff")
    parser.add_argument('-l', '--file_list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-c', '--column', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    argument_dict = args.__dict__
    list_of_files = argument_dict.get('file_list')
    column = argument_dict.get('column')
    main(list_of_files,column)
    # print(list_of_files)
    # print(column)