from bokeh.plotting import curdoc
from visualisation import BokehServer

server = BokehServer(curdoc())
server.run_visualisation()
