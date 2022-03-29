import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pickle
import json
import numpy as np
from problem_description import Problem
from plot_functions import plot_selected_watersheds, \
    plot_selected_contourlines, \
    swc_allocation_create_background_map,\
    swc_allocation_layout, plot_selected_landuse_map

class Solution:

    _id = 0

    def __init__(self, representation, objective_values):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values

dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app = dash_app.server
dash_app.title = "Title to be defined"

def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig

def create_layout():
    # Actual layout of the dash_app
    return html.Div(
        id="root",
        children=[
            html.Div(
                id="header",
                children=[
                    html.H4(children="Header"),
                    html.P(
                        id="description",
                        children="The illustrated Pareto Front shows the optimal solutions."
                    ),
                ],
            ),
            html.Div(
                id="dash_app-container",
                children=[
                    html.Div(
                        id="left-column",
                        children=[
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "SWC allocation",
                                        "value": "benchmark_1",
                                    },
                                    {
                                        "label": "Land use allocation",
                                        "value": "benchmark_2",
                                    },
                                    # {
                                    #     "label": "Benchmark 3",
                                    #     "value": "benchmark_3",
                                    # },
                                ],
                                value="benchmark_1",
                                id="study_area-dropdown",
                            ),
                            dcc.Dropdown(
                                        options=[
                                            {
                                                "label": "Scatter plot",
                                                "value": "scatter",
                                            },
                                        ],
                                        value="scatter",
                                        id="plotmode-dropdown",
                                    ),

                                html.Div(
                                id="pareto_front-container",
                                children=[
                                    html.P(
                                        "Pareto front title",
                                        id="pareto_front-title",
                                    ),
                                    dcc.Graph(
                                        id="pareto_front",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="map-container",
                        children=[
                            html.P(id="layer-selector", children="Plot solutions"),
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Yes",
                                        "value": "Yes",
                                    },
                                    {
                                        "label": "No",
                                        "value": "No",
                                    },
                                ],
                                value="No",
                                id="layer-dropdown",
                            ),
                            #

                            dcc.Graph(
                                id="selected_data",
                                figure= blank_figure()
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

def interactiveParetoFront(dash_app, problems, save_front = None):

    def generate_figure_image(figure, points,layout, opacity):
        figure.add_trace(go.Scatter(
            x=points.iloc[:, 0],
            y=points.iloc[:, 1],
            showlegend=True,
            legendgroup="scatterpoints",
            textposition="top left",
            mode="markers",
            marker=dict(size=3, symbol="circle", opacity=opacity),

        ),
            secondary_y=False,
        )

        figure.update_layout(
            title={
                'y': 0.975,
                'x': 0.055,
                'xanchor': 'left',
                'yanchor': 'top'},
            legend={'itemsizing': 'constant'})

        return figure
    @dash_app.callback(
        Output("pareto_front", "figure"),
        [Input("study_area-dropdown", "value"),]
    )
    def display_scatter_plot(problem_name):
        if problem_name == "benchmark_1":
            problem = problems[0]
        elif problem_name == "benchmark_2":
            problem = problems[1]

        axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)
        layout = go.Layout(
            # title={'text': "Pairwise Pareto front of objectives {} and {}".format("Objective 1",
            #                                                                       "Objective 2"),
            #        'x': 0.2, },
            margin=dict(r=0, l=0, t=0, b=0),
            scene=dict(xaxis=axes, yaxis=axes),
            width=1500, height=700,
            autosize=True,
            xaxis_title=problem.objective_names[0],
            yaxis_title=problem.objective_names[1],
            font=dict(
                family="Arial",
                size=14
            )

        )
        figure = go.Figure(layout=layout)
        figure = make_subplots(figure=figure)

        i = 0
        obj1_values = []
        obj2_values = []
        solution_ids = []
        for pareto_front in problem.benchmarks:
            for solution in pareto_front:
                for realization_id in range(len(solution.objective_values[0])):
                    obj1_values.append(solution.objective_values[0][realization_id])
                    obj2_values.append(solution.objective_values[1][realization_id])
                    solution_ids.append(i)
                i += 1
        scattered_points = pd.DataFrame(
            {'obj1': np.array(obj1_values), 'obj2': np.array(obj2_values), 'sol_id': np.array(solution_ids)})
        plot_mode = "scatter"

        if plot_mode == 'scatter':
            figure = generate_figure_image(figure, scattered_points, layout, opacity=1)

        names = set()
        figure.for_each_trace(
            lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

        #save figure
        if save_front is not None:
            ppi = 96
            width_cm = 3
            height_cm = 2
            pio.write_image(figure, os.path.join(save_front, "pf.svg"),
                            width=(width_cm * 2.54) * ppi,
                            height=(height_cm * 2.54) * ppi)
        return figure

    @dash_app.callback(
        Output("selected_data", "figure"),
        [
            Input("pareto_front", "clickData"),
            Input("study_area-dropdown", "value"),
            Input("layer-dropdown", "value"),
        ],
    )
    ## in brief: get the position of the point in the plot that you click on and find
    # the solution with the corresponding objective values. When its found, select the solution representation
    # and plot it.
    def display_click_image(clickData, problem_name, second_trace = None):
        if problem_name == "benchmark_1":
            problem = problems[0]
        elif problem_name == "benchmark_2":
            problem = problems[1]
        if problem.plot_function_additional_trace is None:
            second_trace = None
        if clickData:
            print(clickData)
            clicked_solution = None
            click_point_np = [float(clickData["points"][0][i]) for i in ["x", "y"]]

            filtered_solutions = []
            for pareto_front in problem.benchmarks:
                for solution in pareto_front:
                    filtered_solutions.append(solution)
                    if click_point_np[0] in solution.objective_values[0] and \
                            click_point_np[1] in solution.objective_values[1]:
                        clicked_solution = solution
            try:
                if clickData and clicked_solution is not None:
                    layout1 = go.Layout(
                        title=f'Corresponding solution',
                        yaxis=dict(showticklabels=False),
                        xaxis=dict(showticklabels=False)
                    )

                    trace1 = problem.plot_function_solution(clicked_solution)

                    # optionable second second trace
                    if second_trace is not None:
                        trace2 = problem.plot_function_additional_trace(clicked_solution)

                    fig = go.Figure(layout=layout1)
                    if problem.plot_layout is not None:
                        fig.update_layout(
                            problem.plot_layout
                        )

                    # add the background dash_app if given as input.
                    # This could be the extent of the study area
                    if problem.plot_background_trace is not None:
                        fig.add_trace(problem.plot_background_trace)

                    fig.add_trace(trace1)
                    #plot the second trace if wanted
                    if second_trace is not None:
                        fig.add_trace(trace2)
                    fig.update_layout(height=700, width=800)
                    return fig
            #if element is not found directly (mouse slightly off) prevent error message
            except  KeyError as error:
                #raise PreventUpdate
                pass
        return {}

# inititalize the dash_app
# dash_app = dash.Dash(
#     __name__,
#     meta_tags=[
#         {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
#     ],
# )

#create the layout
dash_app.layout = create_layout()

#define location of output directory with pareto fronts and required input data for visualizations
filename = 'all_gens_gumobila_200_gens_100_popsize_fixed_index.pkl'

with open(os.path.join("data", filename), 'rb') as handle:
        populations = pickle.load(handle)

with open(os.path.join("data",'watersheds4326.geojson')) as watersheds_json_file:
    watersheds = json.load(watersheds_json_file)

# get the final generation and convert to class solution
final_population =  populations[-1]
final_population_objective_values = [F for F in final_population[0]]
final_population_genes = [X for X in final_population[1]]
optimal_solutions = []
for i in range(len(final_population_objective_values)):
    optimal_solutions.append(Solution(final_population_genes[i],final_population_objective_values[i]))

# get extent of features to define center
lons = []
lats = []
for feature in watersheds["features"]:
    feature_lats = np.array(feature["geometry"]["coordinates"][0][0])[:, 1].tolist()
    feature_lons = np.array(feature["geometry"]["coordinates"][0][0])[:, 0].tolist()
    lons = lons + feature_lons
    lats = lats + feature_lats
mean_lat = np.mean(np.array(lats))
mean_lon = np.mean(np.array(lons))

#create the background (optional), in this case it illustrates the study area extent
background_map = swc_allocation_create_background_map(watersheds)
map_layout = swc_allocation_layout(mean_lat,mean_lon)

swc_allocation_problem = Problem(
    name = "Soil and water conservation measure allocation",
    nr_objectives= 2,
    benchmarks = [optimal_solutions],
    objective_names = ["Minimization of soil loss", "Minimization of labor requirements"],
    objective_descriptions = ["Estimated soil loss computed with RUSLE measured in t/ha/year","Estimated labor requirements in person labor days/ha"],
    objective_functions = [None, None],
    constraint_descriptions = [None],
    validation_functions = [None],
    mathematical_formulation = None,
    plot_layout = map_layout,
    plot_function_solution = plot_selected_watersheds,
    plot_background_trace = background_map,
    plot_geographical_center = [mean_lat, mean_lon],
    plot_function_additional_trace= plot_selected_contourlines)

# #problem 2
with open(r"data\pareto_front.pkl", 'rb') as output:
    pf_comola = pickle.load(output)
optimal_solutions_comola = []
for i in range(len(pf_comola[2])):
    optimal_solutions_comola.append(Solution( np.array(pf_comola[2][i].representation),np.array(pf_comola[2][i].fitness).reshape((4,1))))

comola_problem = Problem(
    name = "Constrained multi-objective land use allocation",
    nr_objectives= 4,
    benchmarks = [optimal_solutions_comola],
    objective_names = ["Maximize Crop Yield", "Maximize Habitat Heterogeneity", "Maximize Forest Species Richness", "Maximize Water Yield"],
    objective_descriptions = [
        "Crop yield basis on number of allocated land of cropland 1-5, where 1 has the lowest intensity and 5 the highest.",
        "Habitat Heterogeneity is computed by summing up the edges of neighboring cells with different land uses and assigning them a weight.",
        "Forest Species richness depends on the total area of forest.",
        "Each land use gets assigned a specific water yield parameter, and Crop land 1 is the optimal land use for this objective."
            ],
    objective_functions = [None, None],
    constraint_descriptions = ["Land use transition constraints: Urban can not be transitioned into another land use. Pasture can only be transitioned into Forest, not the other way.",
                               "Area constraints: Pasture has area constraints of minimum of 10% and maximum of 30% of the total area. Forest has area constraints of minimum of 10% and maximum of 25% of the total area."],
    validation_functions = [None],
    mathematical_formulation = None,
    plot_layout = None,
    plot_function_solution = plot_selected_landuse_map,
    plot_background_trace = None,
    plot_geographical_center = None,
    plot_function_additional_trace= None)

problems = [swc_allocation_problem, comola_problem]

interactiveParetoFront(dash_app, problems)
if __name__ == '__main__':
    dash_app.run_server(debug=True)
