import os, sys, argparse
from graphviz import Digraph


def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def read_code_file(file_path):
    code_lines = {}
    with open(file_path) as fp:
        for ln, line in enumerate(fp):
            assert isinstance(line, str)
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            code_lines[ln + 1] = line
        return code_lines


def filter_nodes_with_code_property(nodes):
    node_ids = []
    node_indices = []
    node_id_to_node = {}
    for node_index, node in enumerate(nodes):
        if 'code' in node.keys() and node['code'].strip() != '':
            id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(id)
            node_id_to_node[id] = {
                'label': node['code'].strip(),
                'node': node
            }
    return node_indices, node_ids, node_id_to_node


def extract_nodes_with_location_info(nodes):
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number
    pass


def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS':  
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES':  
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list


def create_visual_graph(code, adjacency_list, file_name='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose)


def create_graph_from_nodes_and_edges(nodes_ids_to_nodes, edges, allowed_edge_types):
    graph = Digraph('Code Property')
    nodes_to_edges = {}
    for edge in edges:
        edge_type = edge['type'].strip()
        if edge_type in allowed_edge_types.keys():
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id in nodes_ids_to_nodes.keys():
                if start_node_id not in nodes_to_edges:
                    nodes_to_edges[start_node_id] = 0
                nodes_to_edges[start_node_id] += 1
            if end_node_id in nodes_ids_to_nodes.keys():
                if end_node_id not in nodes_to_edges:
                    nodes_to_edges[end_node_id] = 0
                nodes_to_edges[end_node_id] += 1
    for node_id in nodes_to_edges.keys():
        graph.node(name=node_id, label=str(nodes_ids_to_nodes[node_id]['label']))
    for edge in edges:
        edge_type = edge['type'].strip()
        if edge_type in allowed_edge_types.keys():
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id in nodes_ids_to_nodes.keys() and end_node_id in nodes_ids_to_nodes.keys():
                graph.edge(start_node_id, end_node_id, color=allowed_edge_types[edge_type])
    graph.render('Code Dependency File', view=True)
    print(graph)
    pass


def create_forward_slice(adjacency_list, line_no):
    sliced_lines = set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    sliced_lines = sorted(sliced_lines)
    return sliced_lines


def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph


def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph
    pass


def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', help='Name of code file', default='test1.c')
    parser.add_argument('--line', help='Line Number for slice start point', type=int, default=22)
    parser.add_argument('--data_flow_only', action='store_true', help='Slice only on data flow graph.')
    parser.add_argument('--output', help='Output where slice results will be stored.', default='slice-output')
    parser.add_argument('--verbose', help='Show the slice results and the graph.', action='store_true')
    args = parser.parse_args()
    directory = 'tmp'
    file_name = args.code
    slice_ln = int(args.line)
    code_file_path = os.path.join(directory, file_name)
    nodes_path = os.path.join('parsed', directory, file_name, 'nodes.csv')
    edges_path = os.path.join('parsed', directory, file_name, 'edges.csv')
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    code = read_code_file(code_file_path)
    indices, node_ids, node_id_to_node = filter_nodes_with_code_property(nodes)
    allowed_edge_types = {
        'REACHES': 'blue',
    }
    create_graph_from_nodes_and_edges(node_id_to_node, edges=edges, allowed_edge_types=allowed_edge_types)

    pass
