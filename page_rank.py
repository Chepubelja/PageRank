"""
    PageRank algorithm.
    An algorithm used by Google Search to rank websites
    in search engine results.

    Examples of usage:

    python page_rank.py web-Google.txt
    python page_rank.py web-Google.txt --beta 0.85
    python page_rank.py web-Google.txt --epsilon 0.00001
    python page_rank.py web-Google.txt --beta 0.85 --epsilon 0.00001
"""
from time import time
import argparse
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


class PageRank(object):
    """
    PageRank algorithm.
    """
    def __init__(self, file_dir, beta, epsilon):
        """
        Class Constructor.
        """
        self.file_dir = file_dir
        self.beta = beta  # beta = (1 - teleportation probability).
        self.epsilon = epsilon  # epsilon: Minimum needed change value between iterations.
        self.num_nodes = 875713
        self.num_edges = 5105039
        # =========================================
        data = self.read_data()
        old_unique_id, rearranged_data = self.rearrange_id(data)
        adjacency_matrix = self.get_adjacency_matrix(rearranged_data)
        self.power_iteration(adjacency_matrix, old_unique_id)

    def read_data(self):
        """
        Read all text from .txt document.
        :return: All data saved in pandas Dataframe.
        """
        data = pd.read_table(self.file_dir, sep="\t", header=None)
        data.columns = ["FromNodeId", "ToNodeId"]
        return data

    def rearrange_id(self, data):
        """
        Rearranges all id to range: 0 - 875713 (number of nodes)
        :param data: Input data.
        :return: Old unique id, Data with new rearranged id.
        """
        concatenated = np.concatenate((np.array(data['FromNodeId']),
                                       np.array(data['ToNodeId'])))
        old_unique_id = np.unique(concatenated)
        new_unique_id = range(self.num_nodes)

        all_id = pd.DataFrame({
            'FromNodeId': old_unique_id,
            'New_id_from': new_unique_id
        })
        merged_from_node = data.merge(all_id, how='left', on=['FromNodeId'])
        all_id.columns = ['ToNodeId', 'New_id_to']
        rearranged_data = merged_from_node.merge(all_id, how='left', on=['ToNodeId'])
        rearranged_data = rearranged_data.drop(columns=['FromNodeId', 'ToNodeId'])
        rearranged_data.columns = ['FromNodeId', 'ToNodeId']
        return old_unique_id, rearranged_data

    def get_adjacency_matrix(self, rearranged_data):
        """
        Computes sparse adjacency matrix.
        :param rearranged_data: Input data with rearranged id.
        :return: Adjacency matrix.
        """
        data = np.ones(self.num_edges)
        matrix = csr_matrix((data, (rearranged_data['FromNodeId'], rearranged_data['ToNodeId'])),
                            shape=(self.num_nodes, self.num_nodes))
        return matrix

    def power_iteration(self, adjacency_matrix, old_unique_id):
        """
        Power iteration method for finding PageRank values.
        :param adjacency_matrix: Sparse adjacency matrix.
        :param old_unique_id: Old unique node ids.
        :return: Normalized PageRank array.
        """
        convergence = False
        iteration = 0

        degree_with_beta = adjacency_matrix.sum(axis=0).T / self.beta

        # Initializing a vector
        r_0 = np.ones((self.num_nodes, 1)) / self.num_nodes

        while not convergence:

            # Ignoring division by 0
            with np.errstate(divide='ignore'):
                # Calculating the Matrix-by-Vector product r_1
                r_1 = adjacency_matrix.dot(r_0 / degree_with_beta)

            r_1 += (1 - r_1.sum()) / self.num_nodes

            # Convergence condition
            norm = np.linalg.norm(r_0 - r_1, ord=1)
            if norm < self.epsilon:
                convergence = True

            # Reassign vector r_0
            r_0 = r_1
            iteration += 1

            print("Iteration : ", iteration, ' Norm:', norm)

        print("Resulting PageRank array:", r_0)
        print()
        print("Element with the lowest PageRank:", old_unique_id[np.argmin(r_0)])
        print("Element with the highest PageRank:", old_unique_id[np.argmax(r_0)])
        print()


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Process configuration arguments')
    required_args = parser.add_argument_group('REQUIRED arguments')
    required_args.add_argument(action='store', dest='file_dir',
                               type=str, help="Enter path/to/input/file (txt) ")

    optional_args = parser.add_argument_group('OPTIONAL arguments')
    optional_args.add_argument('--beta', action='store', dest='beta',
                               type=float, default=0.9, help="Enter beta: 0.8, 0.9...")
    optional_args.add_argument('--epsilon', action='store', dest='epsilon',
                               type=float, default=0.00001, help="Enter epsilon: 0.01, 0.001...")

    args = parser.parse_args()
    print("===============================================")
    print("Path to txt file: ", args.file_dir)
    print("Beta:             ", args.beta)
    print("Epsilon:          ", args.epsilon)
    print("===============================================")

    start_time = time()

    PageRank(file_dir=args.file_dir, beta=args.beta, epsilon=args.epsilon)

    print("Time of execution:", time() - start_time, "sec")


if __name__ == "__main__":
    main()
