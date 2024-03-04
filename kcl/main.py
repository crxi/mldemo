import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamsync as ss
from functools import lru_cache
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

STEP_0 = 'step0.0'

# df has the following columns:
# - x and y: coordinates of row item
# - cluster: actual cluster in the simulation
# - step0.5, step1.5, step2.5..., stepN.5: cluster allocation
#   at step N of k-clustering algorithm
#   step0.5 corresponds to initial random allocation
#   step1.0 is the creation of centroids
#   step1.5 corresponds to allocation based on step1.0 centroids

def generate_clustered_data(num_clusters=3, total_points=100, seed=None,
                            var=2, radius=9):
    if seed is not None: np.random.seed(seed)

    # Spread the cluster centers around a circle
    angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
    cluster_centers = np.zeros((num_clusters, 2))
    for i, angle in enumerate(angles):
        r = np.random.uniform(3, radius)
        cluster_centers[i, 0] = r * np.cos(angle)
        cluster_centers[i, 1] = r * np.sin(angle)

    # Create datapoints
    points_per_cluster = total_points // num_clusters + 1
    dfs = []
    for k, center in enumerate(cluster_centers):
        points = multivariate_normal.rvs(mean=center, cov=[[var, 0], [0, var]],
                                         size=points_per_cluster)
        d = pd.DataFrame(points, columns=['x', 'y'])
        d['cluster'] = k
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = df.iloc[:total_points]
    df["cluster"] = df["cluster"].astype("category")
    return df
    
def guess_clusters(df, k, k_seed):
    np.random.seed(k_seed)
    a = np.arange(len(df)) % k
    np.random.shuffle(a)
    guess = 'step0.5'
    df[guess] = a
    df[guess] = df[guess].astype("category")
    return df

def compute_centroids(df, cluster):
    return df.groupby(cluster).mean(numeric_only=True).reset_index().iloc[:,:3]

def update_clusters(df, step, centroids):
    distances = cdist(df[['x', 'y']], centroids[['x', 'y']])
    closest_centroids = np.argmin(distances, axis=1)
    df[step] = closest_centroids
    df[step] = df[step].astype("category")

def plot(df, step=STEP_0, centroids=None):
    colors = px.colors.qualitative.Plotly  
    if step == STEP_0:
        fig = px.scatter(df, x='x', y='y', color_discrete_sequence=['black'])
    else:
        # Apply custom color mapping based on the 'step' column
        fig = px.scatter(df.sort_values(by=step), x='x', y='y', color=step,
            color_discrete_map={i: colors[i] for i in range(len(colors))})
    maxy = 14
    fig.update_layout(
        title='Scatter Plot of Data Points',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        xaxis=dict(range=[-maxy, maxy]),
        yaxis=dict(range=[-maxy, maxy]),
        width=700, height=600,
        autosize=True
    )
    fig.update_traces(marker=dict(size=10))
    #for trace in fig.data: trace.showlegend = False

    if centroids is not None:
        # Use the same custom color mapping for centroids
        for i, cent in centroids.iterrows():
            fig.add_trace(go.Scatter(x=[cent['x']], y=[cent['y']], mode='markers',
                    marker=dict(symbol='x', size=15, line=dict(width=2),
                                color=colors[int(cent['cluster'])]),
                    name=f"{int(cent['cluster'])}"))

    fig.update_traces(mode="markers", hovertemplate = None, hoverinfo = "skip")
    fig.update_layout(legend=dict(x=0, y=1, traceorder="normal", title_text='',
         bgcolor="LightSteelBlue", bordercolor="Black", borderwidth=2,
         font=dict(family="sans-serif", size=10, color="black"),
    ))
    return fig

def get(state, keys, typ=str):
    return [typ(state[k.strip()]) for k in keys.split(' ')]

@lru_cache
def do_kclustering(sim_cluster, sim_size, sim_seed, sim_var, k, k_seed, show_actual):
    df = generate_clustered_data(sim_cluster, sim_size, sim_seed, sim_var)
    df = guess_clusters(df, k, k_seed)
    plots = {STEP_0: plot(df, 'cluster' if show_actual else STEP_0)}
    i = 0.5  # Initialize step counter
    centroids = None
    while True:
        stepi = f'step{i:.1f}'
        plots[stepi] = plot(df, stepi, centroids)
        centroids = compute_centroids(df, stepi)
        centroids.columns = ['cluster', 'x', 'y']
        plots[f'step{i+0.5:.1f}'] = plot(df, stepi, centroids)
        new_cluster_col = f'step{i+1:.1f}'
        update_clusters(df, new_cluster_col, centroids)
        if i > 0 and df[stepi].equals(df[new_cluster_col]):
            break
        i += 1  # Increment step counter
    for k,v in plots.items(): v.layout.legend.title = k
    return df, plots 

# StreamSync interface ========================================================
def draw(state):
    sim_seed, sim_cluster, sim_size, k = get(state, 'sim_seed sim_cluster sim_size k', int)
    k_seed, k = get(state, 'k_seed k', int)
    sim_var, k_step = get(state, 'sim_var k_step', float)
    show_actual = 'actual' in state['show']
    _, plots = do_kclustering(sim_cluster, sim_size, sim_seed, sim_var, k, k_seed, show_actual)
    state['k_step_max'] = max_step = max(float(k[4:]) for k in plots.keys() if 'step' in k)
    k_step = min(k_step, max_step)
    key = STEP_0 if k_step < 0 else f'step{k_step:.1f}'
    state['graph'] = plots[key]

# Initialise the state
# "_my_private_element" won't be serialised or sent to the frontend,
# because it starts with an underscore

initial_state = ss.init_state({
    'title': 'K-Cluster Demo',
    'sim_seed': 313,
    'sim_cluster': 5,
    'sim_size': 100,
    'sim_var': 8,
    'k_seed': 31347,
    'k' : 5,
    'k_step': 0.0,
    'show': '',
})

draw(initial_state)