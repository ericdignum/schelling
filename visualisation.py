import time
import logging
import numpy as np
from tornado import gen
from model import Schelling
from functools import partial
from os.path import dirname, join
from concurrent.futures import ThreadPoolExecutor

# Bokeh imports
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.transform import linear_cmap
from bokeh.application import Application
from bokeh.layouts import row, column, gridplot
from bokeh.document import without_document_lock
from bokeh.application.handlers.function import FunctionHandler
from bokeh.models import (ColumnDataSource, Button, Select, Slider, CDSView,
                          BooleanFilter, HoverTool, Div, WheelZoomTool)


class BokehServer():
    """
    Class to run the Bokeh server for visualisation.

    Args:
        doc: a Bokeh Document instance

    Attributes:
        grid_plot: initiates the plot for the grid
        line_plot: initiates one line plot
        update: updates the plots with every time step
        visualise_model: starts the whole visualisation
        grid_values: obtains all the grid values from Mesa.grid
    """

    def __init__(self, doc):
        
        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        self.doc = doc

        # Initialise a Schelling model with the default setting.
        self.model = Schelling()
        self.res_ended = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialise layout and populate the plots with the initial
        # configuration
        self.reset_data()
        self.layout()
        
        print('Visualisation running...')
    
    def init_grid_plot(self):
        """
        Initiates the grid plot for the visualisation.
        """

        # Agent colours, agent tooltips and grid initialisation
        mapper = linear_cmap(field_name='type',
                             palette=['blue', 'orange', 'green', 'red', 'purple'] ,
                             low=0,
                             high=4)
        TOOLTIPS = [('Homophily', '@homophily'),
                    ('Fraction similar', '@fraction_sim'),
                    ('Satisfied', "@satisfied")]
        hover = HoverTool(names=["households"], tooltips=TOOLTIPS)
        self.grid = figure(x_range=(-1,
                                    self.model.width),
                           y_range=(-1,
                                    self.model.grid.height),
                           tools=[hover, 'tap', 'pan',
                                  WheelZoomTool()],
                           tooltips=TOOLTIPS, output_backend="webgl")

        # Plot unsatisfied households
        self.create_filters()
        self.grid.circle(x="x",
                         y="y",
                         radius=0.4,
                         source=self.source,
                         fill_color=mapper,
                         line_color='black',
                         alpha=0.8,
                         view=self.unsatisfied_view,
                         nonselection_fill_alpha=0.2,
                         selection_fill_alpha=1,
                         name='households')
        # Plot satisfied households
        self.grid.rect(x="x",
                         y="y",
                         width=1,
                         height=1,
                         source=self.source,
                         fill_color=mapper,
                         line_color='black',
                         alpha=0.8,
                         view=self.satisfied_view,
                         nonselection_fill_alpha=0.2,
                         selection_fill_alpha=1,
                         name='households')

        self.source.selected.on_change("indices", self.select_households)

    def select_households(self, attr, old, new):
        """
        This function selects all households that are neighbours.

        Args:
            attr: -
            old: -
            new (list): indices that are clicked (can be multiple)
        """
        try:
            index = new[0]
        except IndexError:
            return
        x, y = self.data.iloc[index][['x', 'y']]
        agents = self.model.grid.get_neighbors(
            pos=(x, y), moore=True, radius=self.model.radius)
        positions_set = set([(x,y)])
        _ = [positions_set.add(agent.pos) for agent in agents]

        
        neighbours = []
        coordinates = self.data[['x', 'y']]
        for i, row in coordinates.iterrows():
            if (row['x'], row['y']) in positions_set:
                neighbours.append(i)
        
        self.source.selected.indices = list(neighbours)

    def init_line_plot(self, width=200, height=175, mode='fixed'):
        """
        Initiates the line plot for the server.
        """

        # Create a ColumnDataSource that can be updated at every step.
        TOOLTIPS = [
            ("Average fraction similar neighbours", "@avg_fraction_sim"),
            ("Fraction of satisfied agents", "@satisfied_fraction")
        ]
        self.plot = figure(tooltips=TOOLTIPS,
                           y_range=(0, 1),
                        #    plot_width=width,
                           plot_height=height,
                           sizing_mode=mode,
                           title="Average fraction similar and satisfied",
                           output_backend="webgl")

        for y, color, label in [('avg_fraction_sim', 'purple', "Average fraction similar neighbours"),
                            ('satisfied_fraction', 'green', 'Fraction of satisfied agents'),
                            ('fraction_sat_1', 'blue', 'Fraction satisfied (blue)'),
                            ('fraction_sat_2', 'orange', 'Fraction satisfied (orange)')]:
            self.plot.line(x='time',
                            y=y,
                            source=self.line_source,
                            line_width=5,
                            color=color,
                            legend_label=label)
            self.plot.circle(x='time',
                             y=y,
                             source=self.line_source,
                             size=10,
                             color=color,
                             legend_label=label)
    
        self.plot.legend.location = 'top_left'

    def init_distribution_plot(self, width=200, height=150, mode='fixed'):
        """
        Initiates the distribution plots for residential and school utility.
        """

        self.distribution_plot = figure(title="Homophilies",
                                        x_range=(0, 1),
                                        # plot_width=width,
                                        plot_height=height,
                                        sizing_mode=mode,
                                        output_backend="webgl")

        for group, color in [(0, 'blue'), (1, 'orange')]:
            hist_data = self.data[self.data['type']==group]['homophily']

            # Residential utility
            hist, edges = np.histogram(hist_data,
                                    density=True,
                                    bins=50)
            self.res_quads = self.distribution_plot.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                fill_color=color,
                line_color="white",
                alpha=0.7,
                legend_label='Homophilies ' + color + ' group')

    def init_segregation_plot(self, width=200, height=175, mode='fixed'):
        """
        Initiates the line plot for the server.
        """

        # Create a ColumnDataSource that can be updated at every step.
        TOOLTIPS = [
            ("Segregation 4 neighbourhoods", "@2"),
            ("Segregation 16 neighbourhoods", "@4"),
            ("Segregation 64 neighbourhoods", "@8"),
            ("Segregation 256 neighbourhoods", "@16")
        ]
        self.seg_plot = figure(tooltips=TOOLTIPS,
                           y_range=(0, 1),
                        #    plot_width=width,
                           plot_height=height,
                           sizing_mode=mode,
                           title="Segregation at different levels",
                           output_backend="webgl")

        for y, color, label in [('2', 'teal', "Segregation 4 neighbourhoods"),
                                ('4', 'tan', 'Segregation 16 neighbourhoods'),
                                ('8', 'violet', 'Segregation 64 neighbourhoods'),
                                ('16', 'sienna', 'Segregation 256 neighbourhoods')]:
            self.seg_plot.line(x='time',
                            y=y,
                            source=self.seg_source,
                            line_width=5,
                            color=color,
                            legend_label=label)
            self.seg_plot.circle(x='time',
                             y=y,
                             source=self.seg_source,
                             size=10,
                             color=color,
                             legend_label=label)
    
        self.seg_plot.legend.location = 'top_left'

    def reset_data(self):
        """
        Resets the data, could be the initial reset (new sources need to be
        created) or a subsequent one (only update the data).
        """

        # Callback object to run, step and reset the model properly.
        self.residential = True
        self.callback_obj = None
        self.data, self.system_data = self.model.get_bokeh_vis_data()

        # Check if it's the initial reset (create new sources) or a reset button
        # click (update .data only)
        try:
            self.source.data = self.data
            self.line_source.data = self.system_data
            self.seg_source.data = self.model.calculate_segregation()
            self.update_data()

        # Reset is clicked --> update .data only
        except AttributeError:
            self.source = ColumnDataSource(self.data)
            self.line_source = ColumnDataSource(self.system_data)
            self.seg_source = ColumnDataSource(self.model.calculate_segregation())      
        
    def create_filters(self):

        satisfied_filter = np.array([bool(satisfied) for satisfied in self.source.data['satisfied']])
        unsatisfied_filter = np.invert(satisfied_filter)
        self.satisfied_view = CDSView(source=self.source, 
                    filters=[BooleanFilter(satisfied_filter)])
        self.unsatisfied_view = CDSView(source=self.source, 
                filters=[BooleanFilter(unsatisfied_filter)])

    def update_filters(self):
        """
        Updates the view filters for households, schools and neighbourhoods as
        they can change when reset is clicked (i.e., new model instance).
        """

        satisfied_filter = np.array([bool(satisfied) for satisfied in self.source.data['satisfied']])
        unsatisfied_filter = np.invert(satisfied_filter)

        self.satisfied_view.filters[0] = BooleanFilter(satisfied_filter)
        self.unsatisfied_view.filters[0] = BooleanFilter(unsatisfied_filter)

    def update_data(self):
        """
        Updates all data sources.
        """

        # Update all plots in the figure
        self.data, system_data = self.model.get_bokeh_vis_data()
        self.source.stream(self.data, len(self.data))
        self.line_source.stream(system_data)
        self.seg_source.stream(self.model.calculate_segregation())
        self.update_filters()

    def blocking_task(self):
        time.sleep(0.01)

    @without_document_lock
    @gen.coroutine
    def unlocked_task(self):
        """
        Needed to make sure that if the reset button is clicked it can go
        inbetween events, otherwise it can be quite slow.
        """
        yield self.executor.submit(self.blocking_task)
        self.doc.add_next_tick_callback(partial(self.step_button))

    def run_button(self):
        """
        Handles the run button clicks, coloring and starts the simulation.
        """
        if self.run.label == 'Run':
            self.run.label = 'Stop'
            self.run.button_type = 'danger'
            self.callback_obj = self.doc.add_periodic_callback(self.unlocked_task, 500)

        else:
            self.run.label = 'Run'
            self.run.button_type = 'success'
            self.doc.remove_periodic_callback(self.callback_obj)
            
    def step_button(self):
        """
        Checks which process need to be stepped and execute the step. The
        simulate function of the Model instance cannot be used as we need to
        visualise every step.
        """

        # If residential is not converged yet or below max steps, do a step
        if self.model.max_steps > self.model.schedule.steps:
            self.model.step()
            self.ended = self.model.convergence_check()
        else:
            self.ended = True

        self.update_data()

        # Both processes are done/converged
        if self.ended:
            self.run_button()
            return
        
    def reset_button(self):
        """
        Resets the model and takes the (possible) new parameter values into
        account.
        """

        # Update the parameter values and start a new model
        self.ended = False

        if self.mode.value.lower()=='homogeneous':
            mu2 = float(self.mu1.value)
            std2 = float(self.std1.value)
        else:
            mu2 = float(self.mu2.value)
            std2 = float(self.std2.value)

        self.model = Schelling(
            mode=self.mode.value,
            width=int(self.size.value), 
            height=int(self.size.value), 
            density=float(self.density.value),
            max_steps=int(self.max_steps.value), 
            minority_pc=float(self.minority_pc.value), 
            window_size=int(self.window_size.value), 
            conv_threshold=float(self.conv_threshold.value),
            move_fraction=float(self.move_fraction.value),
            torus=bool(self.torus.value), 
            radius=int(self.radius.value),
            mu1=float(self.mu1.value), 
            std1=float(self.std1.value),
            mu2=mu2, 
            std2=std2
        )

        # Stop the model when it is still running while reset is clicked.
        if self.run.label == 'Stop' and self.callback_obj is not None:
            self.doc.remove_periodic_callback(self.callback_obj)
            self.run.label = 'Run'
            self.run.button_type = 'success'

        self.reset_data()
        self.doc.clear()
        self.layout()

    def layout(self):
        """
        Sets up the whole layout; widgets and all plots.
        """

        # Initialise all plots and widgets
        widgets = self.widgets(width=200)

        plot_width = 500
        sizing_mode = 'scale_width'
        self.init_grid_plot()
        self.init_line_plot(width=plot_width, mode=sizing_mode)
        self.init_distribution_plot(width=plot_width, mode=sizing_mode)
        self.init_segregation_plot(width=plot_width, mode=sizing_mode)

        
        width = 210
        widget_row = column(widgets, width=width)

        desc = Div(text=open(join(dirname(__file__),
                                  "description.html")).read(),
                   margin=0)
        # Column with all the controls and description
        first_col = column(widget_row, width=width, sizing_mode='fixed')

        # Column with the grid/map
        second_col = column([
            row(self.grid, sizing_mode='scale_both'),
        ],
                            sizing_mode='scale_both')

        # Column with the plots
        third_col = column([
            
            desc,
            row(self.buttons(), sizing_mode='scale_width'),
            self.plot, self.seg_plot, self.distribution_plot
        ])

        vis_layout = gridplot([[third_col, second_col, first_col]],
                              toolbar_location=None, sizing_mode='scale_both')

        self.doc.add_root(vis_layout)
        self.doc.title = "Schelling"

    def buttons(self, width=100):
        self.run = Button(label="Run", button_type='success', height=32)
        self.run.on_click(self.run_button)
        self.step = Button(label="Step", button_type='primary', height=32)
        self.step.on_click(self.step_button)
        self.reset = Button(label="Reset", button_type='warning', height=32)
        self.reset.on_click(self.reset_button)
        buttons = [self.run, self.step, self.reset]
        return buttons

    def widgets(self, width=100):
        """
        Hardcodes all widgets.
        """

        header_size = '<h3>'

        # Simulation
        self.max_steps = Slider(start=0,
                                    end=1000,
                                    value=self.model.max_steps,
                                    step=10,
                                    title="Maximum steps",
                                    width=width)
        self.conv_threshold = Select(title="Convergence threshold",
                                options=['0.001', '0.005', '0.01', '0.02'],
                                value=str(self.model.conv_threshold),
                                width=width)
        self.window_size = Slider(start=10,
                                  end=50,
                                  value=self.model.window_size,
                                  step=10,
                                  title="Convergence window size",
                                  width=width)
        
        text = header_size + 'Simulation' + header_size
        simulation_div = Div(text=text, width=width)
        simulation = [simulation_div, self.max_steps,
            self.conv_threshold, self.window_size]

        # Grid
        self.size = Select(title="Size",
                           options=[str(x * 10) for x in range(1, 16)],
                           value=str(self.model.width),
                           width=width)
        self.density = Slider(start=0,
                                        end=1,
                                        value=self.model.density,
                                        step=.05,
                                        title="Density",
                                        width=width)
        self.minority_pc = Slider(start=0,
                                 end=1,
                                 value=self.model.minority_pc,
                                 step=.05,
                                 title="Share blue",
                                 width=width)
        self.move_fraction = Slider(start=0,
                                 end=1,
                                 value=self.model.move_fraction,
                                 step=.05,
                                 title="Fraction moved / step",
                                 width=width)
        self.torus = Select(title="Torus",
                            options=['True', 'False'],
                            value='True',
                            width=width)

        text = header_size + 'Environment' + header_size
        environment_div = Div(text=text, width=width)
        grid = [environment_div, self.size, self.density, 
                self.minority_pc, self.move_fraction] #, self.torus]

        # Households
        self.mode = Select(title="Mode",
                             options=['Heterogeneous', 'Homogeneous'],
                             value=str(self.model.mode),
                             width=width)
        self.radius = Select(title="Radius",
                             options=[str(x) for x in range(1, 11)],
                             value=str(self.model.radius),
                             width=width)
        self.mu1 = Slider(start=0,
                                 end=1,
                                 value=self.model.mu1,
                                 step=.05,
                                 title="Mean (blue)",
                                 width=width)
        self.std1 = Slider(start=0,
                                 end=1,
                                 value=self.model.std1,
                                 step=.01,
                                 title="Std (blue)",
                                 width=width)
        self.mu2 = Slider(start=0,
                                 end=1,
                                 value=self.model.mu2,
                                 step=.1,
                                 title="Mean (orange)",
                                 width=width)
        self.std2 = Slider(start=0,
                                 end=1,
                                 value=self.model.std2,
                                 step=.01,
                                 title="Std (orange)",
                                 width=width)
        
       

        text = header_size + 'Households' + header_size
        household_div = Div(text=text, width=width)
        household = [
            household_div, 
            self.mode,
            self.mu1, self.std1,
            self.mu2, self.std2,
            self.radius
        ]

        widgets = simulation + grid + household

        return widgets

    def run_visualisation(self):
        
        apps = {'/': Application(FunctionHandler(BokehServer))}
        server = Server(apps, port=5004)
        # To avoid bokeh's logger spamming
        log = logging.getLogger('bokeh')
        log.setLevel('CRITICAL')
        import warnings
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

