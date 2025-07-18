import os
import graphviz
import matplotlib.pyplot as plt
import numpy as np

def draw_net(config, genome, view=False, filename=None, show_disabled=True, prune_unused=False, node_names=None, node_colors=None):
    if graphviz is None:
        raise ImportError("This function requires graphviz but it is not installed.")

    from neat.graphs import feed_forward_layers

    node_names = node_names if node_names else {}
    node_colors = node_colors if node_colors else {}

    dot = graphviz.Digraph(format='svg')
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    layers = feed_forward_layers(config.genome_config.input_keys,
                                 config.genome_config.output_keys,
                                 genome.connections)

    used_nodes = set(config.genome_config.input_keys + config.genome_config.output_keys)
    if prune_unused:
        for cg in genome.connections.values():
            if cg.enabled:
                used_nodes.add(cg.key[0])
                used_nodes.add(cg.key[1])
    else:
        used_nodes.update(genome.nodes.keys())

    for n in inputs:
        name = node_names.get(n, str(n))
        dot.node(name, name, shape='box', style='filled', fillcolor=node_colors.get(n, 'lightgray'))

    for n in outputs:
        name = node_names.get(n, str(n))
        dot.node(name, name, shape='box', style='filled', fillcolor=node_colors.get(n, 'lightblue'))

    for layer in layers:
        for n in layer:
            if n in inputs or n in outputs:
                continue
            if n not in used_nodes:
                continue
            name = node_names.get(n, str(n))
            dot.node(name, name, style='filled', fillcolor=node_colors.get(n, 'white'))

    for cg in genome.connections.values():
        if not cg.enabled and not show_disabled:
            continue
        if cg.key[0] not in used_nodes or cg.key[1] not in used_nodes:
            continue
        input_name = node_names.get(cg.key[0], str(cg.key[0]))
        output_name = node_names.get(cg.key[1], str(cg.key[1]))
        style = 'solid' if cg.enabled else 'dotted'
        color = 'green' if cg.weight > 0 else 'red'
        width = str(0.1 + abs(cg.weight / 2.0))
        dot.edge(input_name, output_name, style=style, color=color, penwidth=width)

    if filename:
        dot.render(filename, view=view)
    elif view:
        dot.view()

    return dot

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev = np.array(statistics.get_fitness_stdev())

    plt.figure()
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    if ylog:
        plt.yscale('log')
    plt.grid()
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = {}
    for s in species_sizes[0].keys():
        curves[s] = [species_sizes[g].get(s, 0) for g in range(num_generations)]

    plt.figure()
    for s, sizes in curves.items():
        plt.plot(sizes, label=f"Species {s}")

    plt.title("Speciation over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Size per Species")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)
    if view:
        plt.show()

    plt.close()
