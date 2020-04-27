import pandas as pd

from shapely.geometry.polygon import LinearRing, Polygon
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import DataProcessing
import numpy as np
import time
def generateImages(filtered_graph,data,column):
    filtered_graph = filtered_graph[
        ['level', 'node_x', 'node_y', 'path', 'aggregated_weight', 'actual_weight', 'normalized', 'res_dir_x', 'res_dir_y',
         'res_dir_x_1', 'res_dir_y_1', 'resultant']]

    data['levels'] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    levels = [0.25, 0.5, 0.75]
    fig = plt.figure(figsize=(18, 16), dpi=80)
    ax3 = fig.add_subplot(111)
    # plt.contour(np.array(data['levels']).reshape(699, 639), levels,cmap='cool')
    plt.contourf(np.array(data['levels']).reshape(699, 639), levels, cmap='autumn', alpha=0.7)
    filtered_graph = filtered_graph[filtered_graph['normalized'] >= 0.01]
    filtered_graph_level_0 = filtered_graph[(filtered_graph['level'] == 0)]
    filtered_graph_level_0 = DataProcessing.assignColor(filtered_graph_level_0, 'normalized')
    max_path = filtered_graph_level_0['path'].max()
    color_list = filtered_graph_level_0['color'].tolist()
    start_time_for_creating_contour_paths = time.time()

    for i in range(max_path):
        current_path_x = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['node_x']
        current_path_y = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['node_y']
        if (len(current_path_x) > 15):
            points = np.array([current_path_x, current_path_y]).T
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]
            interpolator = interp1d(distance, points, kind='cubic', axis=0)
            alpha = np.linspace(0, 1, 100)
            current_path_x = pd.Series(interpolator(alpha).T[0])
            current_path_y = pd.Series(interpolator(alpha).T[1])

        list_current_points = list(zip(current_path_x.tolist(), current_path_y.tolist()))
        if((len(list_current_points) >= 3) & ( not (current_path_y.eq(0).any())) & ((current_path_y < 600).all()) & ((current_path_x < 600).all())):
            color = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['color'].tolist()[0]
            ring = LinearRing(list_current_points)
            x, y = ring.xy
            plt.plot(x, y, color=color)
        else:
            if (len(filtered_graph_level_0[filtered_graph_level_0['path'] == i]['color'].tolist()) > 0):
                color = filtered_graph_level_0[filtered_graph_level_0['path'] == i]['color'].tolist()[0]
                plt.plot(current_path_x, current_path_y, 'C3', lw=1, color=color)

    filtered_graph_level_1 = filtered_graph[(filtered_graph['level'] == 1)]
    filtered_graph_level_1 = DataProcessing.assignColor(filtered_graph_level_1, 'normalized')
    max_path = filtered_graph_level_1['path'].max()
    color_list = filtered_graph_level_1['color'].tolist()
    for i in range(max_path):
        current_path_x = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['node_x']
        current_path_y = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['node_y']
        if (len(current_path_x) > 15):
            points = np.array([current_path_x, current_path_y]).T
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]
            interpolator = interp1d(distance, points, kind='cubic', axis=0)
            alpha = np.linspace(0, 1, 100)
            current_path_x = pd.Series(interpolator(alpha).T[0])
            current_path_y = pd.Series(interpolator(alpha).T[1])

        list_current_points = list(zip(current_path_x.tolist(), current_path_y.tolist()))
        if((len(list_current_points) >= 3) & ( not (current_path_y.eq(0).any())) & ((current_path_y < 600).all()) & ((current_path_x < 600).all())):
            color = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['color'].tolist()[0]
            ring = LinearRing(list_current_points)
            x, y = ring.xy
            plt.plot(x, y, color=color)
        else:
            if (len(filtered_graph_level_1[filtered_graph_level_1['path'] == i]['color'].tolist()) > 0):
                color = filtered_graph_level_1[filtered_graph_level_1['path'] == i]['color'].tolist()[0]
                plt.plot(current_path_x, current_path_y, 'C3', lw=1, color=color)

    filtered_graph_level_2 = filtered_graph[(filtered_graph['level'] == 2)]
    filtered_graph_level_2 = DataProcessing.assignColor(filtered_graph_level_2, 'normalized')
    max_path = filtered_graph_level_2['path'].max()
    color_list = filtered_graph_level_2['color'].tolist()
    for i in range(max_path):
        current_path_x = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['node_x']
        current_path_y = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['node_y']
        if (len(current_path_x) > 15):
            points = np.array([current_path_x, current_path_y]).T
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]
            interpolator = interp1d(distance, points, kind='cubic', axis=0)
            alpha = np.linspace(0, 1, 100)
            current_path_x = pd.Series(interpolator(alpha).T[0])
            current_path_y = pd.Series(interpolator(alpha).T[1])

        list_current_points = list(zip(current_path_x.tolist(), current_path_y.tolist()))
        if((len(list_current_points) >= 3) & ( not (current_path_y.eq(0).any())) & ((current_path_y < 600).all()) & ((current_path_x < 600).all())):
            color = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['color'].tolist()[0]
            ring = LinearRing(list_current_points)
            x, y = ring.xy
            plt.plot(x, y, color=color)
        else:
            if (len(filtered_graph_level_2[filtered_graph_level_2['path'] == i]['color'].tolist()) > 0):
                color = filtered_graph_level_2[filtered_graph_level_2['path'] == i]['color'].tolist()[0]
                plt.plot(current_path_x, current_path_y, 'C3', lw=1, color=color)

    print("For creating contour paths %s seconds" % (time.time() - start_time_for_creating_contour_paths))
    start_time_for_quiver_plot = time.time()
    df1 = filtered_graph[filtered_graph['resultant'] >= 0]
    df2 = filtered_graph[filtered_graph['resultant'] <  0]

    # for index, row in filtered_graph.iterrows():
    #     if (row['resultant'] >= 0):
    #         plt.quiver(row['node_x'], row['node_y'], row['res_dir_x_1'], row['res_dir_y_1'],
    #                    width=0.002, headwidth=5.5, headlength=5.5, color='black', scale=500)
    #     else:
    #         plt.quiver(row['node_x'], row['node_y'], row['res_dir_x_1'], row['res_dir_y_1'],
    #                    width=0.002, headwidth=5.5, headlength=5.5, color='blue', scale=500)

    plt.quiver(df1['node_x'], df1['node_y'], df1['res_dir_x_1'], df1['res_dir_y_1'],
                       width=0.002, headwidth=5.5, headlength=5.5, color='black', scale=500)

    plt.quiver(df2['node_x'], df2['node_y'], df2['res_dir_x_1'], df2['res_dir_y_1'],
                       width=0.002, headwidth=5.5, headlength=5.5, color='blue', scale=500)
    # plt.savefig('SMOIS_13_14_15_second.png')
    print("For creating quiver plot %s seconds" % (time.time() - start_time_for_quiver_plot))
    plt.show()