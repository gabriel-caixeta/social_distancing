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
    def __init__(self, category,total_size,recover_period,mortality):
        self.recover_period = recover_period
        self.total_size = total_size
        self.category = category
        self.status = 'Healthy'
        self.position_x = self.total_size*np.random.random_sample()
        self.position_y = self.total_size*np.random.random_sample()
        self.speed = [0,0]
        self.redirect()
        self.sick_days = 0
        self.mortality = mortality

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
                self.end_of_disease()
                # self.recovered()

    def end_of_disease(self):
        if np.random.random_sample() <= self.mortality:
            self.status = 'Dead'
            self.speed = [0,0]
        else:
            self.recovered()

    # randomizes the speed
    def redirect(self):
        self.speed = [(np.random.random_sample() - 0.5) / 5, (np.random.random_sample() - 0.5) / 5]

    # infect the person, if the person is not social spacing, randomizes the speed
    def contaminate(self):
        if (not self.status == 'Sick') and (not self.status == 'Recovered') and (not self.status == 'Dead'):
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
                 recover_period=200,
                 mortality = 0):

        self.recover_period = recover_period
        self.population = int(population)
        self.social_spacing = int(social_spacing)
        self.contamination_rate = float(contamination_rate)
        self.output_filename = output_filename
        self.total_size = int(total_area)
        self.mortality = float(mortality)

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
        self.people = [Person('Normal',self.total_size,self.recover_period,self.mortality) for i in range(self.population)]

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
        self.d_dead, = self.ax.plot([person.position_x for person in self.people if person.status == 'Dead'],
                                    [person.position_y for person in self.people if person.status == 'Dead'],'ko')
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
            self.d_dead.set_data([person.position_x for person in self.people if person.status == 'Dead'],
                                 [person.position_y for person in self.people if person.status == 'Dead'])

            sick = len(self.d_sick.get_data()[0])
            healthy = len(self.d_healthy.get_data()[0])
            recovered = len(self.d_recovered.get_data()[0])
            dead = len(self.d_dead.get_data()[0])

            self.data.append([sick, healthy, recovered,dead])

            print('{} -> Sick: {}, Healthy: {}, Recovered: {}, Dead: {}'.format(i, sick, healthy, recovered, dead))

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
        sick = [d[0] for d in self.data]

        self.data = [[d[0] for d in self.data],
                     [d[1] for d in self.data],
                     [d[2] for d in self.data],
                     [d[3] for d in self.data]]

        print("Peak of sick people: {}".format(max(sick)))

        position = np.arange(len(sick))

        colors = ['red', 'blue', 'yellow','black']
        labels = ['Sick','Recovered','Recovered','Dead']

        prev = [0 for i in self.data[0]]

        for color, data, label in zip(colors,self.data, labels):

            p = plt.bar(position, data,color=color,label=label,bottom=prev, width=1)

            prev = add_lists([prev,data])
        plt.legend(labels)
        plt.xticks([])

        if self.output_filename:
            try:
                output_file = self.output_filename + '_Graph.png'
                plt.savefig(output_file, facecolor='w', edgecolor='w')
            except:
                print("Unable to save the simulation graph")

        plt.show()

def add_lists(args):
    tmp = []
    for ind, item in enumerate(args[0]):
        t = item
        for lst in args[1:]:
            t += lst[ind]
        tmp.append(t)
    return tmp



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--population", help="total population (defauld: 100)")
    parser.add_argument("-s", "--social_spacing", help="Total people on social spacing (default: 0)")
    parser.add_argument("-c", "--contamination_rate", help="contamination rate from 0 to 1 (default: 1)")
    parser.add_argument("-a", "--total_area",
                        help="if specified, will change the size of the environment. Used to crowd or open the space (default: 10)")
    parser.add_argument("-r", "--recover_period",
                        help="if specified, will change how long it takes for a sick person to recover. (default: 200 moves)")
    parser.add_argument("-o", "--output_filename",nargs='?',const='experiment', help="if specified, will output the mp4 and graph. Note: either output or show the simulation real time")
    parser.add_argument("-m", "--mortality", help="if specified, will change the chance of death(0-1). (default: 0)")
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
    if args.mortality:
        params.update({"mortality": args.mortality})

    experiment = SocialDistancing(**params)
    experiment.run_simulation()
    experiment.plot_graph()
