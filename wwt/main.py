import numpy as np
import plotly.graph_objs as go
import streamsync as ss

# we only need to create these once
x = np.linspace(-3, 4, 141)
y = np.linspace(-3, 4, 141)
X, Y = np.meshgrid(x, y)

def _get(state, *key): return (state[k] for k in key)

def plot_line(state):
    a, b, z = _get(state['abz'], 'a', 'b', 'z')
    La, Lb, Lx, Ly, Lz = _get(state['label'], 'a', 'b', 'x', 'y', 'z')
    Z = a * X + b * Y    
    
    if 'contours' in state['options']:    
        labels = dict(start=-10, end=10, size=0.5, showlabels = True,
                      labelfont = dict(size = 14, color = 'black'))
    else:
        labels = dict(start=z, end=z, size=0.5)
    if 'heatmap' in state['options']:
        colorscale = [(0,'#0000ff'), (0.5,'#9999ff'), (0.5,'#ff9999'),(1.0,'#ff0000')]
    else:
        colorscale = [(0.0, '#ffffff'), (1.0, '#ffffff')]
    templ = (f'{La} {Lx} + {Lb} {Ly}'
             f'<br>  = {a:.2f}' + ' &times; %{x:.1f} + ' f'{b:.2f}' ' &times; %{y:.1f}'
              '<br>  = %{z:.3f}<extra></extra>')
    zabs = 2
    contour = go.Contour(
        x=x, y=y, z=Z, 
        colorscale=colorscale, zmin=z-zabs, zmax=z+zabs,
        contours=dict(coloring='heatmap', **labels),
        opacity=0.6, showscale=False, hovertemplate=templ,
    )
    data = [contour]

    font=dict(size=24, color='#0000FF', family='Arial, sans-serif')
    axislo = dict(range=[-1.1, 2.1], fixedrange=True, constrain='domain', dtick=1, tick0=0,
            zeroline=True, zerolinewidth=2, zerolinecolor='#AAAAFF',
            title_font=font)
    layout = go.Layout(
        title=f'{a:.3f} {Lx} + {b:.3f} {Ly} = {z}', title_font=font,
        width=700, height=700, paper_bgcolor='#EEEEEE',
        xaxis=dict(title=Lx, **axislo, scaleanchor='y', scaleratio=1),
        yaxis=dict(title=Ly, **axislo),
        margin=dict(l=30, r=30, t=50, b=30),
    )
    fig = go.Figure(data=data, layout=layout)

    if 'ttab' in state['options']:
        truth = state['truth']
        for i in [0,1]:
            for j in [0,1]:
                _add_truth_table(fig, i, j, truth[i,j], state)
    
    state['graph'] = fig

    def inv(v): return 'inf' if np.isclose(v,0) else f'{z/v:.3f}'
    msg = f'''
    NOTES
    - {Lx}-axis intercept = {Lz}/{La} = {inv(a)}
    - {Ly}-axis intercept = {Lz}/{Lb} = {inv(b)}
    '''
    state['msg'] = msg

def _add_truth_table(fig, x, y, truth, state):
    a, b = _get(state['abz'], 'a', 'b')
    La, Lb, Lx, Ly = _get(state['label'], 'a', 'b', 'x', 'y')

    value = a * x + b * y
    templ = (f'{La} {Lx} + {Lb} {Ly}'
            f'<br>  = {a:.2f}' + ' &times; %{x:.1f} + ' f'{b:.2f}' ' &times; %{y:.1f}'
            f'<br>  = {value:.3f}<extra></extra>')
    # Determine color based on truth
    circle_color = 'black' if truth == 0 else 'red'
    text_color = 'pink' if truth == 0 else 'white'
    # Add a scatter plot for the circle
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', showlegend=False,
                             marker=dict(color=circle_color, size=28),
                             hoverinfo='none', hovertemplate=templ))
    
    # Add annotation (label) inside the circle
    fig.add_annotation(x=x, y=y, text=str(truth), showarrow=False,
                       font=dict(color=text_color, size=16),
                       xanchor='center', yanchor='middle')
    return fig
    

def reset(state):
    state['abz'] = defabz
    state['truth'] = np.zeros(shape=(2,2), dtype='int32')
    plot_line(state)

def minus1(state):
    s = state['abz']
    s['a'] *= -1
    s['b'] *= -1
    s['z'] *= -1
    plot_line(state)

def click(state, payload):
    c = payload[0]
    if 1 <= c['curveNumber'] <= 4:
        x, y = c['x'], c['y']
        v = state['truth'][x,y]
        state['truth'][x,y] = 0 if v else 1
        plot_line(state)

def options_changed(state, payload):
    state['options'] = payload
    state['label'] = labelwwt if 'usewwt' in payload else labelabz
    plot_line(state)


labelwwt = dict(a='w0', b='w1', x='x0', y='x1', z='θ',
                title='w0⋅x0 + w1⋅x1 = θ')
labelabz = dict(a='a', b='b', x='x', y='y', z='z',
                title='a⋅x + b⋅y = z')
deflabel = labelabz
defabz = dict(a=-1,b=1,z=0)

initial_state = ss.init_state({
    'label': deflabel,
    'abz': defabz,
    'truth': np.zeros(shape=(2,2), dtype='int32'),
    'graph' : None,
    'options': [],
    'msg': '',
})

plot_line(initial_state)
