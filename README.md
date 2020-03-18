# Social Distancing

A simulation on how social distancing can decrease the speed at which a disease spreads.

## Logic

### Assumptions on people who are actively engaged on social distancing:

- they don't go out much (static dots on the simulation)
- if they do get in touch with people and end up infected with the disease, they do not pass the disease on to others


### 



## How to run the simulation

### Necessary libraries

- numpy
- matplotlib
- pandas
- ffmpeg

### Parameters

| Input | Title | Definition | Default |
|---|----|----|---|
| -p | population | Total population |100|
| -s | social_spacing | Total people on social spacing |0|
| -c | contamination_rate | Possibility of spreading the disease when interacting with someone |1|
| -a | total_area | The size of the environment |20|
| -r | recover_period | How long it takes for a person to get cured (in movements) |200
| -o | output_filename | The output file ('_SimulationVideo.mp4' and '_Graph.png' will be added)|Experiment|


### Example

#### Command

<pre><code>python main.py -p 150 -s 50 -c 0.5 -a 25 -o Simulation1</code></pre>

#### Output

- Simulation1_SimulationVideo.mp4:![Simulation Example](https://imgur.com/5hxtm4E)
- Simulation1_Graph.png:![Graph Example](https://imgur.com/qHLtHMK)