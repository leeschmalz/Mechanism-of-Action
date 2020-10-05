def create_network(max_parents, max_children, connections):
    from pgmpy.models import BayesianModel

    '''
    Takes in max parents / chilldren per node constraints and network definition (connections). 
    Outputs model and list of nodes included in network.
    '''
    
    model = BayesianModel([connections[0]])
    nodes_added = []
    
    for edge in connections[1:]:
            #whichever has more children 
            #Node A
            try:
                if edge[0] in nodes_added and edge[1] in nodes_added: #both already in network
                    if len(model.get_children(edge[0])) < max_children and len(model.get_children(edge[1])) < max_children:
                        if len(model.get_parents(edge[0])) < max_parents and len(model.get_parents(edge[1])) < max_parents:
                            add_first = random.choice([0,1])
                            model.add_edge(edge[add_first],edge[1-add_first])

                        if len(model.get_parents(edge[0])) < max_parents and len(model.get_parents(edge[1])) >= max_parents:
                            model.add_edge(edge[1],edge[0])

                        if len(model.get_parents(edge[0])) >= max_parents and len(model.get_parents(edge[1])) < max_parents:
                            model.add_edge(edge[0],edge[1])

                    if len(model.get_children(edge[0])) < max_children and len(model.get_children(edge[1])) >= max_children and len(model.get_parents(edge[1])) < max_parents:
                        model.add_edge(edge[0],edge[1])

                    if len(model.get_children(edge[0])) >= max_children and len(model.get_children(edge[1])) < max_children and len(model.get_parents(edge[0])) < max_parents:
                        model.add_edge(edge[1],edge[0])

                elif edge[0] in nodes_added and edge[1] not in nodes_added: #Node A in network Node B not in network
                    if len(model.get_children(edge[0])) < max_children:
                        model.add_edge(edge[0],edge[1])
                        nodes_added.append(edge[1])

                elif edge[0] not in nodes_added and edge[1] in nodes_added: #Node A not in network Node B in network
                    if len(model.get_children(edge[1])) < max_children:
                        model.add_edge(edge[1],edge[0])
                        nodes_added.append(edge[0])

                else:
                    #neither in network, choose randomly
                    add_first = random.choice([0,1])
                    model.add_edge(edge[add_first],edge[1-add_first])
                    nodes_added.append(edge[add_first])
                    nodes_added.append(edge[1-add_first])

            except ValueError: #catch non-DAG error
                pass

    #if node is leaf, connect to target
    for node in nodes_added:
        if node in model.get_leaves():
            model.add_edge(node,'target')
            
    return model, nodes_added