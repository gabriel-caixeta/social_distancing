import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pandas as pd
import argparse

plt.style.use('seaborn-pastel')

CONTAMINATION_AREA = 0.5


# check if there was an interaction between two people
# interaction is defined by a proximity of less than the contamination area defined above
def interaction(person_a, person_b):
    global CONTAMINATION_AREA
    if abs(person_a.position_x - person_b.position_x) <= CONTAMINATION_AREA:
        if abs(person_a.position_y - person_b.position_y) <= CONTAMINATION_AREA:
            return True
    return False


'''
class that defines a person with the following attributes:
recover_period: how long it takes for the person to recover. (constant for every person)
total_size: total area that the person can move in
category: 'Normal' or 'SocialSpacing'
status: 'Healthy', 'Sick' or 'Recovered'
position_x/y: position of the person on the grid
speed: (x,y) vector that defines the direction and speed at which is moving
sick_days: how long the person has been sick
'''
class Person:
    def __init__(self, category,total_size,recover_period):
        self.recover_period = recover_period
        self.total_size = total_size
        self.category = category
        self.status = 'Healthy'
        self.position_x = self.total_size*np.random.random_sample()
        self.position_y = self.total_size*np.random.random_sample()
        self.speed = [0,0]
        self.redirect()
        self.sick_days = 0

    def stop(self):
        self.speed = [0,0]

    def social_space(self):
        self.category = 'SocialSpacing'
        self.speed = [0,0]

    def is_social_spacing(self):
        return True if self.category == 'SocialSpacing' else False

    def update_position(self):
        self.position_x = self.position_x + self.speed[0]
        self.position_y = self.position_y + self.speed[1]

        # graph edges
        if self.position_x >= self.total_size:
            self.position_x = self.total_size
            self.speed[0] = -1 * self.speed[0]
        if self.position_x <= 0:
            self.position_x = 0
            self.speed[0] = -1 * self.speed[0]
        if self.position_y >= self.total_size:
            self.position_y = self.total_size
            self.speed[1] = -1 * self.speed[1]
        if self.position_y <= 0:
            self.position_y = 0
            self.speed[1] = -1 * self.speed[1]

        if self.status == 'Sick':
            self.sick_days += 1

            if self.sick_days >= self.recover_period:
                self.recovered()

    # randomizes the speed
    def redirect(self):
        self.speed = [(np.random.random_sample() - 0.5) / 5, (np.random.random_sample() - 0.5) / 5]

    # infect the person, if the person is not social spacing, randomizes the speed
    def contaminate(self):
        if (not self.status == 'Sick') and (not self.status == 'Recovered'):
            self.status = 'Sick'
            self.sick_days = 0
            if not self.is_social_spacing():
                self.redirect()

    def recovered(self):
        self.status = 'Recovered'

    def is_contaminated(self):
        return True if self.status == 'Sick' else False


'''
Main function for 
'''
class SocialDistancing:
    def __init__(self,
                 population=100,
                 social_spacing=0,
                 output_filename=None,
                 contamination_rate=1,
                 total_area=20,
                 recover_period=200):

        self.recover_period = recover_period
        self.population = int(population)
        self.social_spacing = int(social_spacing)
        self.contamination_rate = float(contamination_rate)
        self.output_filename = output_filename
        self.total_size = int(total_area)

        self.people = []

    def check_contamination(self,person):
        for person_a in self.people:
            if person == person_a:
                continue
            if interaction(person_a, person):
                if person_a.is_contaminated() and (not person_a.is_social_spacing()):
                    return True
        return False

    def contamination(self):
        for person in self.people:
            if self.check_contamination(person):
                tmp = np.random.random_sample()
                if tmp <= self.contamination_rate:
                    person.contaminate()

    def run_simulation(self):
        self.people = [Person('Normal',self.total_size,self.recover_period) for i in range(self.population)]

        if self.social_spacing:
            people_ss = self.people[-self.social_spacing:]

            for person in self.people:
                if person in people_ss:
                    person.social_space()

        self.people[0].contaminate()

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, self.total_size), ylim=(0, self.total_size))
        plt.xticks([])
        plt.yticks([])

        self.d_sick, = self.ax.plot([person.position_x for person in self.people if person.status == 'Sick'],
                         [person.position_y for person in self.people if person.status == 'Sick'],'ro')
        self.d_recovered, = self.ax.plot([person.position_x for person in self.people if person.status == 'Recovered'],
                         [person.position_y for person in self.people if person.status == 'Recovered'],'yo')
        self.d_healthy, = self.ax.plot([person.position_x for person in self.people if person.status == 'Healthy'],
                         [person.position_y for person in self.people if person.status == 'Healthy'],'bo')
        self.d = [self.d_sick, self.d_healthy, self.d_recovered]

        self.data = []

        self.end_animation = False

        def animate(i):
            # self.people = contamination(self.people,self.contamination_rate)
            self.contamination()
            for person in self.people:
                person.update_position()

            if self.end_animation:
                plt.close()

            if len([x for x in self.people if x.is_contaminated()]) == 0:
                for person in self.people:
                    person.stop()
                self.end_animation = True

            self.d_sick.set_data([person.position_x for person in self.people if person.status == 'Sick'],
                                 [person.position_y for person in self.people if person.status == 'Sick'])
            self.d_recovered.set_data([person.position_x for person in self.people if person.status == 'Recovered'],
                                      [person.position_y for person in self.people if person.status == 'Recovered'],)
            self.d_healthy.set_data([person.position_x for person in self.people if person.status == 'Healthy'],
                                    [person.position_y for person in self.people if person.status == 'Healthy'])

            sick = len(self.d_sick.get_data()[0])
            healthy = len(self.d_healthy.get_data()[0])
            recovered = len(self.d_recovered.get_data()[0])

            self.data.append([sick, healthy, recovered])

            print('{} -> Sick: {}, Healthy: {}, Recovered: {}'.format(i, sick, healthy, recovered))

            return self.d,

        anim = FuncAnimation(self.fig, animate, frames=10000, interval=20)

        if self.output_filename:
            mywriter = animation.FFMpegWriter(fps=60)
            try:
                output_file = self.output_filename+'_SimulationVideo.mp4'
                anim.save(output_file, writer=mywriter)
            except:
                print('An error occurred while trying to save the simulation video')

        plt.show()

    def plot_graph(self):
        def overlapped_bar(df, show=False, width=0.9, alpha=.5,
                           title='', xlabel='', ylabel='', **plot_kwargs):
            N = len(df)
            M = len(df.columns)
            indices = np.arange(N)
            colors = ['yellow', 'blue', 'red']
            for i, label, color in zip(range(M), df.columns, colors):
                kwargs = plot_kwargs
                kwargs.update({'color': color, 'label': label})
                plt.bar(indices, df[label], width=width, alpha=alpha if i else 1, **kwargs)
                plt.xticks([])
            plt.legend()
            plt.title(title)

            plt.ylabel(ylabel)

            if self.output_filename:
                try:
                    output_file = self.output_filename + '_Graph.png'
                    plt.savefig(output_file,facecolor='w', edgecolor='w')
                except:
                    print("Unable to save the simulation graph")
            if show:
                plt.show()
            return plt.gcf()

        sick, healthy, recovered = [],[],[]
        for i,j,k in self.data:
            sick.append(i)
            healthy.append(j)
            recovered.append(k)

        print("Peak of sick people: {}".format(max(sick)))

        df = pd.DataFrame(np.matrix([recovered,healthy, sick]).T, columns=['Recovered', 'Healthy', 'Sick'],
                          index=pd.Index(['T%s' % i for i in range(len(sick))]))
        overlapped_bar(df, show=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--population", help="total population (defauld: 100)")
    parser.add_argument("-o", "--output_filename",nargs='?',const='experiment', help="if specified, will output the mp4 and graph. Note: either output or show the simulation real time")
    parser.add_argument("-c", "--contamination_rate", help="contamination rate from 0 to 1 (default: 1)")
    parser.add_argument("-s", "--social_spacing", help="Total people on social spacing (default: 0)")
    parser.add_argument("-a", "--total_area", help="if specified, will change the size of the environment. Used to crowd or open the space (default: 10)")
    parser.add_argument("-m", "--recover_period", help="if specified, will change how long it takes for a sick person to recover. (default: 200 moves)")
    args = parser.parse_args()
    params = {}
    if args.population:
        params.update({"population":args.population})
    if args.social_spacing:
        params.update({"social_spacing": args.social_spacing})
    if args.output_filename:
        params.update({"output_filename":args.output_filename})
    if args.contamination_rate:
        params.update({"contamination_rate":args.contamination_rate})
    if args.total_area:
        params.update({"total_area": args.total_area})
    if args.recover_period:
        params.update({"recover_period": args.recover_period})

    experiment = SocialDistancing(**params)
    experiment.run_simulation()
    experiment.plot_graph()
