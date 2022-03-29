import numpy as np
import plotly.graph_objs as go
import os
import json
import pickle

def plot_selected_watersheds(clicked_solution):
    with open(os.path.join("data", 'watersheds4326.geojson')) as watersheds_json_file:
        watersheds = json.load(watersheds_json_file)
    ws_lons = []
    ws_lats = []
    indizes_protected_areas = tuple(np.where(clicked_solution.representation == True)[0]+1)
    print(indizes_protected_areas)

    for feature in watersheds["features"]:
        if feature["properties"]["pos_rank"] in indizes_protected_areas:
            for i in feature["geometry"]["coordinates"]:
                for j in i:
                    ws_feature_lats = np.array(j)[:, 1].tolist()
                    ws_feature_lons = np.array(j)[:, 0].tolist()
                    ws_lons = ws_lons + ws_feature_lons + [None]
                    ws_lats = ws_lats + ws_feature_lats + [None]
    ws_lats = np.array(ws_lats)
    ws_lons = np.array(ws_lons)

    # this is the main trace of your solution
    trace = go.Scattermapbox(
        lat=ws_lats,
        lon=ws_lons,
        mode="lines",
        fill="toself",
        #opacity = 0.7,
        fillcolor='rgba(31,120,180,0.6)',
        line=dict(width=1, color = 'rgb(31,120,180)'),
        showlegend=False

    )
    return trace

def plot_selected_contourlines(clicked_solution):
    with open(os.path.join("data",'contourlines4326.geojson')) as terraces_json_file:
        terraces = json.load(terraces_json_file)
    cl_lons = []
    cl_lats = []
    indizes_protected_areas = tuple(np.where(clicked_solution.representation == True)[0] + 1)
    flist = []
    for feature in terraces["features"]:
        if feature["properties"]["pos_rank"] in indizes_protected_areas:
            for f in feature["geometry"]["coordinates"]:
                flist.append(np.array(f))
    for f in flist:
        cl_feature_lats = f[:, 1].tolist()
        cl_feature_lons = f[:, 0].tolist()
        cl_lons = cl_lons + cl_feature_lons + [None]
        cl_lats = cl_lats + cl_feature_lats + [None]
    cl_lats = np.array(cl_lats)
    cl_lons = np.array(cl_lons)
    trace = go.Scattermapbox(
        lat=cl_lats,
        lon=cl_lons,
        mode="lines",
        line=dict(width=0.01, color='rgba(5,5,5, 1.0)'),
        showlegend=False
    )
    return trace

## function to produce the background of your solution map
def swc_allocation_create_background_map(watersheds):
    lons = []
    lats = []
    for feature in watersheds["features"]:
        for i in feature["geometry"]["coordinates"]:
            for j in i:
                feature_lats = np.array(j)[:, 1].tolist()
                feature_lons = np.array(j)[:, 0].tolist()
                lons = lons + feature_lons + [None]
                lats = lats + feature_lats + [None]
    lats = np.array(lats)
    lons = np.array(lons)
    background_map = go.Scattermapbox(
        fill="toself",
        lat=lats,
        lon=lons,
        fillcolor='rgba(27,158,119,0.3)',
        marker={'size': 0, 'color': "#f9ba00"},
        showlegend=False)
    return background_map

def swc_allocation_layout(lat, lon):
    return {"margin": dict(t=5, r=5, b=5, l=5),
     "autosize": True,
     "mapbox":go.layout.Mapbox(
        style="stamen-terrain",
        zoom=12,
        center_lat=lat,
        center_lon=lon,
    )}

def plot_selected_landuse_map(clicked_solution):
    def load_patch_ID_map(filepath):
        return np.genfromtxt(filepath, dtype=int, skip_header=6, filling_values='-1')

    def read_patch_ID_map(patchmap, solution_represenation, static_element, No_Data_Value=0, input_patch_map=None):
        if input_patch_map is None:
            patches = load_patch_ID_map(patchmap)
        else:
            patches = input_patch_map
        landuseraster = []
        counter = 0
        for rowid in range(patches.shape[0]):
            colvalues = []
            for i in range(patches.shape[1]):
                if patches[rowid, i] == No_Data_Value:
                    colvalues.append(static_element)
                else:
                    colvalues.append(solution_represenation[counter])
                    counter += 1

            landuseraster.append(colvalues)
        reversed_lum = np.flip(np.array(landuseraster), axis=0)
        return reversed_lum

    with open(r"data\patch_map.pkl",
              'rb') as output:
        patch_map = pickle.load(output)

    landusemap = read_patch_ID_map(
        patchmap=patch_map,
        solution_represenation=clicked_solution.representation,
        static_element=8,
        input_patch_map=patch_map)

    ##change from here

    trace1 = go.Heatmap(
        z=landusemap,
        colorscale=[
            [0, "rgb(255, 226, 146)"],
            [0.125, "rgb(255, 226, 146)"],
            [0.125, "rgb(240, 230, 135)"],
            [0.25, "rgb(240, 230, 135)"],
            [0.25, "rgb(255, 216, 1)"],
            [0.375, "rgb(255, 216, 1)"],
            [0.375, "rgb(238, 200, 0)"],
            [0.5, "rgb(238, 200, 0)"],
            [0.5, "rgb(208, 173, 2)"],
            [0.625, "rgb(208, 173, 2)"],
            [0.625, "rgb(27, 137, 29)"],
            [0.75, "rgb(27, 137, 29)"],
            [0.75, "rgb(162, 204, 90)"],
            [0.875, "rgb(162, 204, 90)"],
            [0.875, "rgb(105, 105, 105)"],
            [1, "rgb(105, 105, 105)"]
        ],
        colorbar=dict(
            tickvals=[1, 2, 3, 4, 5, 6, 7, 8],
            ticktext=["Cropland 1", "Cropland 2", "Cropland 3",
                      "Cropland 4", "Cropland 5", "Forest", "Pasture", "Urban"]
        )
    )

    return trace1


