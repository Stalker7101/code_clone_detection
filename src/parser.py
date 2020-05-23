from typing import Dict
from queue import Queue

import networkx as nx
from matplotlib import pyplot as plt
from tree_sitter import Language, Parser, Node

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


JAVA_SO_PATH: str = "../data/java.so"


class TreeSitterNode(object):

    def __init__(self, node: Node, program: str = None):
        """
        :param node: The tree_sitter node
        :param program: the str of the program
        """

        self.type = node.type
        self.start_byte = node.start_byte
        self.end_byte = node.end_byte
        self.name = self.get_name(node, program)

    def get_name(self, node: Node, program: str = None) -> str:
        if program is None:
            return node.type

        if 'identifier' in node.type and node.is_named:
            return program[self.start_byte:self.end_byte]
        else:
            return node.type

    def __eq__(self, obj):
        return self.type == obj.type and self.start_byte == obj.start_byte and self.end_byte == obj.end_byte

    def __str__(self):
        return f'{self.name} @ [{self.start_byte}, {self.end_byte}]'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())



def get_parser(so_path: str = None) -> Parser:
    if so_path is None:
        so_path = JAVA_SO_PATH

    JAVA_LANGUAGE = Language(so_path, 'java')

    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    return parser


def parse_program(program: str, parser: Parser = None) -> nx.DiGraph:
    if parser is None:
        parser: Parser = get_parser()

    tree = parser.parse(bytes(program, "utf8"))

    g: nx.DiGraph = nx.DiGraph()

    queue: Queue = Queue()
    queue.put(tree.root_node)

    while not queue.empty():

        node = queue.get()

        if not hasattr(node, 'children'):
            continue

        for child in node.children:
            g.add_edge(TreeSitterNode(node, program), TreeSitterNode(child, program))

            queue.put(child)

    return g


def plot_graph(g: nx.DiGraph):
    from networkx.drawing.nx_agraph import graphviz_layout

    pos = graphviz_layout(g, prog='dot')
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))
    labels: Dict[TreeSitterNode, str] = {node: node.name for node in g.nodes}
    nx.draw(g, pos, ax=ax, with_labels=True, labels=labels, arrows=True, font_size=15, node_color="yellow")


def get_data_from_graph(g:nx.DiGraph) -> Data:
    # graph_spicy = nx.to_scipy_sparse_matrix(g, format='coo')
    # edge_index = torch.tensor([graph_spicy.row, graph_spicy.col], dtype=torch.long)
    # x =  torch.tensor(graph_spicy.data, dtype=torch.float)
    #
    # data = Data(x=x,edge_index=edge_index)
    return from_networkx(g)

def program_to_data(program: str, parser: Parser = None) -> Data:
    return get_data_from_graph(parse_program(program, parser))

if __name__ == '__main__':
    program_str = """
        public int add(int a, int b) {
            int c = 0;
            c = a + b;
            return c;
        }
    """

    program_str_2 = """
        public int add_numbers(int c, int d) {
            int e = 0;
            return c + d;
        }
    """

    parser = get_parser(JAVA_SO_PATH)
    g1 = parse_program(program_str, parser)
    g2 = parse_program(program_str_2, parser)
    g3 = nx.compose(g1, g2)
    plot_graph(g3)
    plt.show()
