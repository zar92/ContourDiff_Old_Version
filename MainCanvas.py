from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        # self.axes.transData.transform_point([x,y])

    # def onclick(event):
    #     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #           ('double' if event.dblclick else 'single', event.button,
    #            event.x, event.y, event.xdata, event.ydata))
    #
    # cid = FigureCanvas.mpl_connect(onclick('button_press_event'))

    def spaghettiPlot(self):
        import pandas as pd
        import numpy as np
        data = pd.read_csv('data0.csv')['SMOIS']
        self.axes.contour(np.array(data).reshape(699, 639))
        self.draw()

    def filledContour(self):

        data = pd.read_csv('data0.csv')['SMOIS']
        self.axes.contourf(np.array(data).reshape(699, 639))
        self.draw()

    def clearPlt(self):
        self.fig.clear()
        self.axes = self.figure.add_subplot(111)
        self.draw()
    def clearPlt2(self):
        self.fig.clear()
        self.axes = self.figure.add_subplot(111)
    def plot_contour(self,data,level):
        self.axes.contour(np.array(data['levels']).reshape(699, 639), level, colors=['g', 'r', 'y'])
        self.draw()
    def generate_images(self,filtered_graph,data,levels,column,alpha_cf = 0.7,flag_dir = 0,flag_content = 0, flag_fillcontours = 1, magnitude = 0,
                        cmap='Colormap 1',cline='copper',line_opacity = 0.4,line_width=1.5):
        cmap_dict =  {'Colormap 1':['#7fc97f','#beaed4','#fdc086','#ffff99'],
        'Colormap 2':['#1b9e77','#d95f02','#7570b3','#e7298a'],
        'Colormap 3':['#a6cee3','#1f78b4','#b2df8a','#33a02c'],
        'Colormap 4':['#e41a1c','#377eb8','#4daf4a','#984ea3'],
        'Colormap 5' :['#66c2a5','#fc8d62','#8da0cb','#e78ac3'],
        'Colormap 6':['#8dd3c7','#ffffb3','#bebada','#fb8072']}
        # cvector_high2low_dict = {'black': '#000000', 'blue': '#0c86f1', 'green': '#136a0f', 'red': ' #e31414'}
        self.clearPlt2()
        filtered_graph = filtered_graph[
            ['level', 'node_x', 'node_y', 'path', 'aggregated_weight', 'actual_weight', 'normalized', 'res_dir_x',
             'res_dir_y',
             'res_dir_x_1', 'res_dir_y_1', 'resultant','mag']]

        data['levels'] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        self.axes.contour(np.array(data['levels']).reshape(699, 639), levels, cmap=cline, alpha=line_opacity,linewidths=line_width)
        if(flag_content == 0 or flag_content == 1):
            self.axes.contourf(np.array(data['levels']).reshape(699, 639), levels, colors=cmap_dict[cmap], alpha=alpha_cf)
            # if(flag_fillcontour == 1):
            #     self.axes.contourf(np.array(data['levels']).reshape(699, 639), levels, colors=cmap_dict[cmap], alpha=alpha_cf)
            # if (flag_fillcontour == -1):
            #     self.filledContour()
        filtered_graph = filtered_graph[filtered_graph['normalized'] >= 0.01]
        df1 = filtered_graph[(filtered_graph['resultant'] >= 0) & (filtered_graph['mag'] > magnitude)]
        df2 = filtered_graph[(filtered_graph['resultant'] < 0) & (filtered_graph['mag'] > magnitude)]
        if(flag_content == 0 or flag_content == 2):
            if(flag_dir == 0 or flag_dir==1):
                self.axes.quiver(df1['node_x'], df1['node_y'], df1['res_dir_x_1'], df1['res_dir_y_1'],
                   width=0.0009, headwidth=5.5, headlength=5.5, color='black', scale=1000)
            if(flag_dir == 0 or flag_dir==2):
                self.axes.quiver(df2['node_x'], df2['node_y'], df2['res_dir_x_1'], df2['res_dir_y_1'],
                   width=0.0009, headwidth=5.5, headlength=5.5, color='blue', scale=1000)
        self.draw()

