from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data-analysis')
def data_analysis():
    return render_template('data_analysis.html')

@app.route('/world-map')
def world_map():
    return render_template('world_map.html')

@app.route('/algo-rendering')
def algo_rendering():
    return render_template('algo_rendering.html')

if __name__ == '__main__':
    app.run(debug=True)
