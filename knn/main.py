import numpy as np
import plotly.graph_objs as go
import streamsync as ss

class KNN:
    def __init__(self, seed=0, maxsize=100):
        L, R, step = 0, 10, 0.1
        self.xg = np.arange(L, R+step, step)  # x axis of grid
        self.yg = np.arange(L, R+step, step)  # y axis of grid
        
        # generate training data
        np.random.seed(seed)
        x = np.round(np.random.uniform(L+step, R-step, maxsize), 4)
        y = np.round(np.random.uniform(L+step, R-step, maxsize), 4)
        v = (x + y > R).astype(int)  # categorize using x + y > R
        self.data = np.vstack([x, y, v]).T
    
    @staticmethod
    def find_nn(xs, ys, data, k):
        # find K nearest neighbours (without loops)      
        xx, yy = np.meshgrid(xs, ys)            # shape is X x Y
        dx = xx[:,:,np.newaxis] - data[:,0]     # shape is X x Y x size
        dy = yy[:,:,np.newaxis] - data[:,1]     # shape is X x Y x size
        dd = dx**2 + dy**2                      # calculate distance squared
        di = np.argsort(dd, axis=2)             # sort and get the indices
        di = di[:,:,:k]                         # we only need K NN's indices
        nn = data[di]                           # select all K NN
        return xx, yy, nn
        
    def simulate(self, size, k, flip):
        #flip = int(round(flip * size / 100))    # number of data to flip
        data = self.data[:size].copy()          # use only 'size' data
        data[:flip,2] = 1 - (data[:flip,2])     # flip category
        xx, yy, nn = KNN.find_nn(self.xg, self.yg, data, k)
        ca = nn[:,:,:,2]                        # select category only
        ca = (ca.sum(axis=2)>k//2).astype(int)  # sum up for majority vote  
        self.category = ca
        self.simdata = data


# StreamSync interface ========================================================
COLOR = {0:'#4444FF', 1:'#FF4444'}
def draw_graph(state):
    if not state['seed']: state['seed'] = '0'
    seed, size, k, flippct = [int(state[i]) for i in 'seed size k flippct'.split()]
    state['flip'] = flip = int(flippct * size/100)

    knn = KNN(seed)
    knn.simulate(size, k, flip)
    xs, ys, vs = knn.simdata.T
    
    trace_0 = go.Scatter(
        x=xs[vs==0], y=ys[vs==0], mode='markers', name='B', hoverinfo='none',
        marker=dict(color=COLOR[0], symbol='circle', size=10),
    )
    trace_1 = go.Scatter(
        x=xs[vs==1], y=ys[vs==1], mode='markers', name='R', hoverinfo='none',
        marker=dict(color=COLOR[1], symbol='x', size=10),
    )
    contour = go.Contour(
        x=knn.xg, y=knn.yg, z=knn.category,
        opacity=0.2, showscale=False,
        line=dict(width=3, color='yellow', dash='dot'),
        colorscale=list(COLOR.items()),
        contours=dict(type="levels", start=0, end=1, size=2),
        hovertemplate='(%{x:.1f}, %{y:.1f})<extra></extra>'
    )
    data = [contour, trace_0, trace_1]
    layout = go.Layout(
        width=700,height=700,
        hovermode='closest', hoverdistance=1,
        xaxis=dict(title='Feature 1', range=[0, 10], fixedrange=True,
          constrain="domain", scaleanchor="y",scaleratio=1),
        yaxis=dict(title='Feature 2', range=[0, 10], fixedrange=True,
          constrain="domain"),
        paper_bgcolor='#EEEEEE',
        margin=dict(l=30, r=30, t=30, b=30),
    )
    # add evaluation result if graph was clicked
    r = ['# Click in graph to select a point for evaluation.']
    click = state['click']
    if click is not None: add_nn(data, layout, r, knn, k, click)    
    state['graph'] = go.Figure(data=data, layout=layout)
    state['result'] = r[0]
    
def graph_click(state, payload):
    p = payload[0]
    state['click'] = [p['x'], p['y']]
    draw_graph(state)

def add_nn(data, layout, result, knn, k, click):
    cx, cy = click
    _, _, nn = knn.find_nn(cx, cy, knn.simdata, k)
    nn = nn[0,0]
    for x,y,v in nn:
        trace = go.Scatter(x=[cx,x],y=[cy,y], mode='lines', showlegend=False,
              line=dict(color=COLOR[int(v)], dash='dot'), hoverinfo='none')
        data.append(trace)
    x,y,v = nn[-1]
    d = np.sqrt((cx-x)**2 + (cy-y)**2)
    count1 = np.sum(nn[:,2] == 1)
    cat = (count1 > k//2).astype(int)
    layout['shapes'] = [go.layout.Shape(type="circle", xref="x", yref="y",
             x0=cx-d, y0=cy-d, x1=cx+d, y1=cy+d,
             line_color=COLOR[cat], fillcolor=COLOR[cat], opacity=0.3
         )]
    msg = f'# Point ({cx:0.1f},{cy:0.1f}) has {k-count1} "B" and {count1} "R" nearest neighbours. So it is "{"BR"[cat]}".' 
    result[0] = msg


# Initialise the state
# "_my_private_element" won't be serialised or sent to the frontend,
# because it starts with an underscore

initial_state = ss.init_state({
    "title": "KNN Demo",
    "size": 20,
    "k": 3,
    "flip": 0,
    "flippct": 0,
    "seed": 313,
    "result": None, 
    "graph" : None,
    'click' : None,
})

draw_graph(initial_state)