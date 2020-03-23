# Social Distancing

A simulation on how social distancing can decrease the speed at which a disease spreads.

## Logic

### Assumptions on people who are actively engaged on social distancing:

- they don't go out much (static dots on the simulation)
- if they do get in touch with people and end up infected with the disease, they do not pass the disease on to others


### Assumptions on awareness

- A person that is aware of the disease, might change their behaviour once infected
- It is directly correlated with the amount of sick and dead people in the same environment

For the purpose of this simulation, implemented the following:

- A person who is infected has a chance of starting social distancing
- Probability = (3*number_of_dead + number_of_currently_sick) / total_population



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
| -t | total_area | The size of the environment |20|
| -a | awareness | If specified, implements the awareness logic | False |
| -r | recover_period | How long it takes for a person to get cured (in movements) |200
| -m | mortality_rate | The chance of death at the end of the disease | 0 |
| -o | output_filename | The output file ('_SimulationVideo.mp4' and '_Graph.png' will be added)|Experiment|


### Example

#### Command

<pre><code>python main.py -p 150 -s 50 -c 0.5 -t 25 -m 0.5 -o Simulation1</code></pre>

#### Output

- Simulation1_SimulationVideo.mp4
  - ![Video Example](/img/Simulation1_Gif.gif)

- Simulation1_Graph.png:
  - ![Graph Example](/img/Simulation1_Graph.png)