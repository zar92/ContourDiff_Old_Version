import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import time
from QuadTree import QTree

def importData(filename):
    """Read data from csv file and transform it into dataframe"""
    data = pd.read_csv(filename)
    return data


def createQuantile(data, column_name, cut_of_point):
    """Create quantile based on the cut of point(0.2,0.5,0.75)"""
    data[data[column_name] > data[column_name].quantile(cut_of_point)] = 0
    return data


def modelTheGraph(contourset):
    """Model the graph as dataframe from contourset"""
    cntr_data = pd.DataFrame(columns=['level', 'node_x', 'node_y', 'path'])
    frames = list()
    start_time_model_graph = time.time()
    for level_index in range(len(contourset.collections)):
        path_counter = 0
        indices = np.arange(0, len(contourset.collections[level_index].get_paths()))
        array_list = np.take(contourset.collections[level_index].get_paths(), indices)
        for item in array_list.flat:
            node_x = item.vertices[:, 0].tolist()
            node_y = item.vertices[:, 1].tolist()
            frames.append([level_index, node_x, node_y, path_counter])
            path_counter += 1
    df = pd.DataFrame(frames, columns=['level', 'node_x', 'node_y', 'path'])
    df1 = df[['level', 'node_y', 'path']]
    df2 = df[['level', 'node_x', 'path']]
    lst_col = 'node_y'
    r1 = pd.DataFrame({
        col: np.repeat(df1[col].values, df1[lst_col].str.len())
        for col in df1.columns.drop(lst_col)}
    ).assign(**{lst_col: np.concatenate(df1[lst_col].values)})[df1.columns]
    lst_col2 = 'node_x'
    r2 = pd.DataFrame({
        col: np.repeat(df2[col].values, df2[lst_col2].str.len())
        for col in df2.columns.drop(lst_col2)}
    ).assign(**{lst_col2: np.concatenate(df2[lst_col2].values)})[df2.columns]
    cntr_data['level'] = r1['level'].tolist()
    cntr_data['node_x'] = r2['node_x'].tolist()
    cntr_data['node_y'] = r1['node_y'].tolist()
    cntr_data['path'] = r1['path'].tolist()
    print("For modeling the graph %s seconds" % (time.time() - start_time_model_graph))

    return cntr_data


def dir_mag_by_5(filelist, column_name):
    """Calculate scalar diffrence of an entry againist its 24 neigbors"""
    mag_list = list()
    dir_list = list()
    for i in range(len(filelist)):
        padded_matrix_1 = np.pad(pd.read_csv(filelist[i])[column_name].values.reshape(699, 639), [(1, 1), (1, 1)],
                                 mode='constant', constant_values=0)  # to calculate 1st degree neighbors
        direction = np.zeros((699, 639, 24))
        org = padded_matrix_1[1:-1, 1:-1]
        dirs = {'dir_0': padded_matrix_1[1:-1, 2:], 'dir_1': padded_matrix_1[0:-2, 2:],
                'dir_2': padded_matrix_1[0:-2, 1:-1],
                'dir_3': padded_matrix_1[0:-2, 0:-2], 'dir_4': padded_matrix_1[1:-1, 0:-2],
                'dir_5': padded_matrix_1[2:, 0:-2],
                'dir_6': padded_matrix_1[2:, 1:-1], 'dir_7': padded_matrix_1[2:, 2:]}
        padded_matrix_2 = np.pad(pd.read_csv(filelist[i])[column_name].values.reshape(699, 639), [(2, 2), (2, 2)],
                                 mode='constant', constant_values=0)  # to calculate 2nd degree neighbors
        dirs2 = {'dir_8': padded_matrix_2[2:-2, 4:], 'dir_9': padded_matrix_2[0:-4, 4:],
                 'dir_10': padded_matrix_2[0:-4, 2:-2],
                 'dir_11': padded_matrix_2[0:-4, 0:-4], 'dir_12': padded_matrix_2[2:-2, 0:-4],
                 'dir_13': padded_matrix_2[4:, 0:-4],
                 'dir_14': padded_matrix_2[4:, 2:-2], 'dir_15': padded_matrix_2[4:, 4:],
                 'dir_16': padded_matrix_2[3:-1, 4:],
                 'dir_17': padded_matrix_2[4:, 3:-1], 'dir_18': padded_matrix_2[4:, 1:-3],
                 'dir_19': padded_matrix_2[3:-1, 0:-4], 'dir_20': padded_matrix_2[1:-3, 0:-4],
                 'dir_21': padded_matrix_2[0:-4, 3:-1], 'dir_22': padded_matrix_2[0:-4, 1:-3],
                 'dir_23': padded_matrix_2[1:-3, 4:]
                 }
        start_time_dir_mag = time.time()
        for i in range(24):
            if (i < 8):
                direction[:, :, i] = org - dirs['dir_' + str(i)]
            else:
                direction[:, :, i] = org - dirs2['dir_' + str(i)]
            mag_list.append(np.linalg.norm(direction, axis=2))
            dir_list.append(direction)
        print("For computing direction and magnitude %s seconds" %(time.time() - start_time_dir_mag))
        return mag_list, dir_list


def draw_dirs2(filelist, column_name):
    """Calculate resultant direction in x and y"""
    mag, direction = dir_mag_by_5(filelist, column_name)
    res_dir_x_list = list()
    res_dir_y_list = list()
    start_time_vector_components = time.time()
    for i in range(len(filelist)):
        res_dir_x = np.zeros_like(mag[i])
        res_dir_y = np.zeros_like(mag[i])
        for d in range(24):
            if (d < 16):
                res_dir_x += direction[i][:, :, d] * np.cos(np.pi / 4 * (d % 8))
                res_dir_y += direction[i][:, :, d] * np.sin(np.pi / 4 * (d % 8))
            else:
                res_dir_x += direction[i][:, :, d] * np.cos(np.pi / 8 * 2 * (d % 8) + 1)
                res_dir_y += direction[i][:, :, d] * np.sin(np.pi / 8 * 2 * (d % 8) + 1)
                res_dir_x_list.append(res_dir_x)
                res_dir_y_list.append(res_dir_y)
    print("For computing vector components %s seconds" % (time.time() - start_time_vector_components))

    return res_dir_x_list, res_dir_y_list, mag, direction


def fetch_direction(file_list, column_name):

    """Aggregate resultant direction in"""
    start_time_aggrigating_vector_components = time.time()
    res_dir_x_list, res_dir_y_list, mag, direction = draw_dirs2(file_list, column_name)
    all_in_x = np.zeros_like(mag[0])
    all_in_y = np.zeros_like(mag[0])
    all_mag = np.zeros_like(mag[0])
    for i in range(len(res_dir_x_list)):
        all_in_x = np.add(all_in_x, res_dir_x_list[i])
        all_in_y = np.add(all_in_y, res_dir_y_list[i])
    res_x = all_in_x.ravel()
    res_y = all_in_y.ravel()
    data = pd.read_csv(file_list[0])
    data['res_x'] = res_x
    data['res_y'] = res_y
    print("For aggregating vector components %s seconds" % (time.time() - start_time_aggrigating_vector_components))
    return data


def createWeightedGraph(contourdf, file_list, column_name):
    """Created a weighted graph from extracted contour"""
    start_time_creating_weighted_graph = time.time()
    weights = np.full((len(contourdf)), 1)  # initialize weights to one
    contourdf['weights'] = weights
    # group the dataframe to count path_length(number of nodes in the path)
    path_length_df = contourdf.groupby(['level', 'path']).size().reset_index(name='path_length')
    path_length_1_df = path_length_df[path_length_df['path_length'] == 1]
    cntr_data_weight_0 = contourdf[(np.isin(contourdf['level'], path_length_1_df['level'])) &
                                   (np.isin(contourdf['path'], path_length_1_df['path']))]
    cntr_data_weight_0['weights'] = 0
    cntr_data__weight_1 = contourdf[~(np.isin(contourdf['level'], path_length_1_df['level'])) |
                                    ~(np.isin(contourdf['path'], path_length_1_df['path']))]

    cntr_data_weight_1_diffrence = (cntr_data__weight_1.shift() - cntr_data__weight_1)
    cntr_data_weight_1_diffrence['calculated_weight'] = (np.sqrt(
        (cntr_data_weight_1_diffrence['node_x'].values) ** 2 + (
            cntr_data_weight_1_diffrence['node_y'].values) ** 2).tolist())

    cntr_data__weight_1['calculated_weight'] = cntr_data_weight_1_diffrence['calculated_weight'].tolist()
    cntr_data__weight_1['path_diff'] = cntr_data_weight_1_diffrence['path'].tolist()
    weight_list = cntr_data__weight_1['calculated_weight'].tolist()
    # for index,row in cntr_data__weight_1.iterrows():
    #     if(row['path_diff'] != 0):
    #         weight_list[index] = weight_list[index + 1]
    indices = cntr_data__weight_1.loc[cntr_data__weight_1['path_diff'] != 0]
    for index, row in indices.iterrows():
        weight_list[index] = weight_list[index + 1]
    cntr_data__weight_1['act2'] = weight_list
    cntr_data__weight_1['actual_weight'] = weight_list
    cntr_data__weight_1 = cntr_data__weight_1[['level', 'node_x', 'node_y', 'path', 'actual_weight']]
    cntr_data_weight_0['actual_weight'] = cntr_data_weight_0['weights']
    cntr_data_weight_0 = cntr_data_weight_0[['level', 'node_x', 'node_y', 'path', 'actual_weight']]
    weighted_df = pd.concat([cntr_data_weight_0, cntr_data__weight_1])
    weighted_df = weighted_df.sort_values(['level', 'path'])
    weighted_df['aggregated_weight'] = weighted_df.groupby(['level', 'path'])['actual_weight'].transform('sum')
    weighted_df = weighted_df[['level', 'node_x', 'node_y', 'path', 'aggregated_weight', 'actual_weight']]
    weighted_df['normalized'] = (weighted_df['aggregated_weight'] - weighted_df['aggregated_weight'].min()) / (
                weighted_df['aggregated_weight'].max() - weighted_df['aggregated_weight'].min())

    data = fetch_direction(file_list, column_name)

    data['node_x_1'] = data['longitude']
    data['node_y_1'] = data['latitude']

    weighted_df['node_x_1'] = weighted_df['node_x'] // 1
    weighted_df['node_y_1'] = weighted_df['node_y'] // 1

    merged_df = weighted_df.merge(data, how='left')
    merged_df = merged_df[['res_x', 'res_y', 'node_x_1', 'node_y_1']]

    weighted_df['res_dir_x'] = merged_df['res_x'].tolist()
    weighted_df['res_dir_y'] = merged_df['res_y'].tolist()

    weighted_df['res_dir_x_1'] = weighted_df['res_dir_x'] * weighted_df['actual_weight']
    weighted_df['res_dir_y_1'] = weighted_df['res_dir_y'] * weighted_df['actual_weight']

    weighted_df['res_dir_x_1'] = weighted_df.groupby(['level', 'path'])['res_dir_x_1'].transform('sum') / weighted_df[
        'aggregated_weight']
    weighted_df['res_dir_y_1'] = weighted_df.groupby(['level', 'path'])['res_dir_y_1'].transform('sum') / weighted_df[
        'aggregated_weight']

    weighted_df['resultant'] = weighted_df['res_dir_x_1'] + weighted_df['res_dir_y_1']
    weighted_df['mag'] = np.sqrt(np.square(weighted_df['res_dir_x_1']) + np.square(weighted_df['res_dir_y_1']))

    print("For creating a weighted graph %s seconds" % (time.time() - start_time_creating_weighted_graph))

    return weighted_df


def filterBasedOnGrid(depth, weighted_graph):
    """Grid"""
    points = list(zip(weighted_graph['node_x'].tolist(), weighted_graph['node_y'].tolist()))
    qtree = QTree(depth, points)
    print(points)
    qtree.subdivide()
    print('I am also here')
    h = qtree.graph()

    x_grid = np.unique(np.array(h[0])//1).tolist()
    y_grid =  np.unique(np.array(h[1])//1).tolist()
    weighted_graph = weighted_graph[
        (np.isin(weighted_graph['node_x'], x_grid)) | (np.isin(weighted_graph['node_y'], y_grid))]
    # print(x_grid)
    return weighted_graph


def assignColor(data, column_name):
    color_1 = data[data[column_name] > data[column_name].median()]
    color_2 = data[data[column_name] <= data[column_name].median()]

    green_list = np.empty(len(color_1), dtype=object)
    green_list[:] = 'green'
    blue_list = np.empty(len(color_2), dtype=object)
    blue_list[:] = 'lime'
    color_1['color'] = green_list.tolist()
    color_2['color'] = blue_list.tolist()
    colored_df = pd.concat([color_1, color_2])
    colored_df = colored_df.sort_values(['level', 'path'])
    return colored_df



