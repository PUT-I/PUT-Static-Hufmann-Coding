""" This script handles encoding and decoding of UTF-8 files with Huffman coding """

import argparse
import multiprocessing
import os
from multiprocessing.pool import Pool, ApplyResult
from typing import List, Dict, Tuple

import numpy as np
from console_progressbar import ProgressBar

from src.text_entropy.calculate_text_entropy import count_symbols_in_string


class Counter:
    """ This class stores int value to be passed by reference in functions """

    def __init__(self, int_value: int = 0):
        """ Initializes Counter class with int value

        :param int_value: initial value of counter
        """

        self._value = int_value

    def get(self) -> int:
        """ Gets current counter value and increments it

        :return counter value before increment
        """

        value_ = self._value
        self._value += 1
        return value_

    def increase(self, inc_value: int) -> None:
        """ Increments counter value

        :param inc_value: value to be added to counter
        """

        self._value += inc_value

    def __repr__(self):
        """ Handles representation of Counter class instances """

        return str(self._value)


class HuffmanNode:
    """ This class represents single node of Huffman tree """

    def __init__(self, priority: float, left=None, right=None, symbol: str = ""):
        """ Initializes HuffmanNode class instance

        :param priority:
        :param left: left child of node
        :param right: right child of node
        :param symbol:
        """

        self.priority: float = priority
        self.left = left
        self.right = right
        self.symbol: str = symbol

    def __eq__(self, other):
        """ Handles comparison (equal) of HuffmanNode class instances

        :param other: HuffmanNode class instance to be compared to
        """

        return self.priority == other.priority

    def __lt__(self, other):
        """ Handles comparison (lesser) of HuffmanNode class instances

        :param other: HuffmanNode class instance to be compared to
        """

        return self.priority < other.priority \
               or (self.priority == other.priority and self.symbol < other.symbol)

    def __repr__(self):
        """ Handles representation of HuffmanNode class instances """

        return f'HuffmanNode(priority = {self.priority}, symbol = "{self.symbol}", ' \
               f'hasLeft = {self.left is not None}, hasRight = {self.right is not None})'


class HuffmanTree:
    """ This class represents Huffman tree """

    def __init__(self, root: HuffmanNode):
        """ Initializes HuffmanTree class instance

        :param root: root node of Huffman tree
        """

        self.root = root

    def encode(self):
        """ Encodes Huffman tree to binary form (filled with 0 to full bytes)

        :return encoded Huffman tree
        """

        encoded = self._encode_node(self.root)
        padding = np.zeros(shape=(8 - len(encoded) % 8,), dtype=np.uint8)
        encoded = np.append(encoded, padding)
        return np.packbits(encoded).tobytes()

    @staticmethod
    def decode(buffer: bytes):
        """ Decodes Huffman tree from binary form

        :param buffer: bytes to be decoded into huffman tree
        :return decoded Huffman tree
        """

        array: np.array = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))
        return HuffmanTree(HuffmanTree._decode_node(array, Counter()))

    @staticmethod
    def _encode_node(node: HuffmanNode, bits: np.array = np.empty(shape=(0,), dtype=np.uint8)) -> np.array:
        """ Recursively encodes Huffman tree node

        :param node: Huffman tree node
        :param bits: bits of already encoded Huffman tree nodes
        :return encoded Huffman tree node
        """

        if node.left is None and node.right is None:
            bits = np.append(bits, HuffmanCoder.UINT8_1)

            symbol_bits = np.unpackbits(np.frombuffer(node.symbol.encode(), dtype=np.uint8))
            bits = np.append(bits, symbol_bits)
        else:
            bits = np.append(bits, HuffmanCoder.UINT8_0)
            if node.left is not None:
                bits = HuffmanTree._encode_node(node.left, bits)
            if node.right is not None:
                bits = HuffmanTree._encode_node(node.right, bits)

        return bits

    @staticmethod
    def _decode_node(bits: np.array, counter: Counter) -> HuffmanNode:
        """ Recursively decodes Huffman tree node

        :param bits: bits of encoded Huffman tree
        :param counter: counter (passed by reference) which indicates current index of bits array
        :return decoded Huffman tree node
        """

        index = counter.get()
        bit = bits[index]
        if bit == np.uint(1):
            symbol_bits: np.array = bits[index + 1:index + 9]
            counter.increase(8)
            symbol = np.packbits(symbol_bits).tobytes().decode()
            return HuffmanNode(0, symbol=symbol)
        else:
            left_node = HuffmanTree._decode_node(bits, counter)
            right_node = HuffmanTree._decode_node(bits, counter)
            return HuffmanNode(0, left_node, right_node)


class HuffmanCoder:
    """ This class provides Huffman encoding and decoding functionality """

    UINT8_0 = np.uint8(0)
    """ Constant value 0 of type uint8 """

    UINT8_1 = np.uint8(1)
    """ Constant value 1 of type uint8 """

    @staticmethod
    def encode_file(in_file_path: str, out_file_path: str, block_size: int = 1024 * 1024) -> None:
        """ Encodes file with Huffman coding

        :param in_file_path: path to input file
        :param out_file_path: path to output file
        :param block_size: size of blocks to be used in encoding
        """

        with open(out_file_path, mode="wb") as output_file, open(in_file_path, mode="r") as input_file:
            output_file.write(bytes())

            frequencies: Dict[str, int] = {}
            while True:
                data = input_file.read(block_size)
                if not data:
                    break

                frequencies_part = count_symbols_in_string(data)
                for key in frequencies_part:
                    if key in frequencies:
                        frequencies[key] = frequencies[key] + frequencies_part[key]
                    else:
                        frequencies[key] = frequencies_part[key]
            input_file.seek(0)

            tree: HuffmanTree = HuffmanCoder._create_tree(frequencies)

            encoded_tree = tree.encode()
            encoded_tree_length_bytes = len(encoded_tree).to_bytes(4, byteorder="big", signed=False)

            output_file.write(encoded_tree_length_bytes)
            output_file.write(encoded_tree)

            file_stats = os.stat(in_file_path)
            data_size = file_stats.st_size
            pb = ProgressBar(total=data_size, prefix='Progress', suffix='finished', decimals=1, length=50, fill='X',
                             zfill='-')
            data_processed = 0
            while True:
                pb.print_progress_bar(data_processed)
                data = input_file.read(block_size)
                if not data:
                    break

                data_processed += len(data)

                encoded, padding_length = HuffmanCoder._encode_data(data, tree)
                data_length_bytes = len(encoded).to_bytes(4, byteorder="big", signed=False)
                padding_length_bytes = padding_length.to_bytes(1, byteorder="big", signed=False)

                block = data_length_bytes + padding_length_bytes + encoded
                output_file.write(block)

            pb.print_progress_bar(data_size)

    @staticmethod
    def decode_file(in_file_path: str, out_file_path: str) -> None:
        """ Encodes file with Huffman coding

        :param in_file_path: path to input file
        :param out_file_path: path to output file
        """

        with open(in_file_path, mode="rb") as input_file, open(out_file_path, mode="w") as output_file:
            output_file.write("")

            tree_length = int.from_bytes(input_file.read(4), byteorder="big", signed=False)  # In bytes
            tree: HuffmanTree = HuffmanTree.decode(input_file.read(tree_length))

            file_stats = os.stat(in_file_path)
            data_size = file_stats.st_size
            pb = ProgressBar(total=data_size, prefix='Progress', suffix='finished',
                             decimals=1, length=50, fill='X', zfill='-')
            data_processed = 4 + tree_length

            pool = Pool(processes=multiprocessing.cpu_count())
            pool_jobs: List[Tuple[ApplyResult, int]] = []
            finish = False
            while not finish:
                # Read enough data to fill pool jobs
                while len(pool_jobs) < multiprocessing.cpu_count():
                    pb.print_progress_bar(data_processed)

                    data_length = int.from_bytes(input_file.read(4), byteorder="big", signed=False)  # In bytes
                    padding_length = int.from_bytes(input_file.read(1), byteorder="big", signed=False)  # In bits

                    data_bytes = input_file.read(data_length)

                    if data_bytes == bytes():
                        finish = True
                        break

                    data_processed += 4 + 1

                    pool_jobs.append(
                        (pool.apply_async(HuffmanCoder._decode_data, (data_bytes, tree, padding_length)), data_length)
                    )

                # Wait for data to be processed
                for job in pool_jobs:
                    decoded = job[0].get()
                    output_file.write(decoded)
                    data_processed += job[1]
                    pb.print_progress_bar(data_processed)
                pool_jobs.clear()

    @staticmethod
    def _create_tree(frequencies: Dict[str, float]) -> HuffmanTree:
        """ Encodes file with Huffman coding

        :param frequencies: dictionary of symbol frequencies
        :return Huffman tree
        """

        nodes: List[HuffmanNode] = []
        for symbol in frequencies:
            nodes.append(HuffmanNode(frequencies[symbol], symbol=symbol))

        nodes = sorted(nodes)

        while len(nodes) > 1:  # 2. While there is more than one node
            left: HuffmanNode = nodes.pop(0)
            right: HuffmanNode = nodes.pop(0)  # 2a. remove two highest nodes
            priority = round(left.priority + right.priority, 7)

            node = HuffmanNode(priority, left, right)  # 2b. create internal node with children
            nodes.append(node)  # 2c. add new node to queue
            nodes = sorted(nodes)

        return HuffmanTree(nodes[0])  # 3. tree is complete - return root node

    @staticmethod
    def _node_to_dict(node: HuffmanNode, prefix: str = "", code_dict: dict = None) -> dict:
        """ Converts Huffman tree node to dicitonary

        :param node: Huffman tree node
        :param prefix: bits that had been already red
        :param code_dict: incomplete code dictionary
        :return Huffman tree transformed to dictionary (key - symbol, value - Huffman code)
        """

        if code_dict is None:
            code_dict = {}

        if node.left is not None:
            HuffmanCoder._node_to_dict(node.left, prefix + "0", code_dict)
        else:
            code_dict[node.symbol] = prefix

        if node.right is not None:
            HuffmanCoder._node_to_dict(node.right, prefix + "1", code_dict)
        else:
            code_dict[node.symbol] = prefix

        return code_dict

    @staticmethod
    def _tree_to_dict(tree: HuffmanTree):
        """ Converts Huffman tree to dictionary

        :param tree: Huffman tree
        :return Huffman tree transformed to dictionary (key - symbol, value - Huffman code)
        """

        return HuffmanCoder._node_to_dict(tree.root)

    @staticmethod
    def _encode_data(data: str, tree: HuffmanTree) -> Tuple[bytes, int]:
        """ Encodes string data with Huffman coding

        :param data: UTF-8 string data
        :param tree: Huffman tree
        :return encoded data and length of padding (in bits)
        """

        code = HuffmanCoder._tree_to_dict(tree)
        code_numpy = {}

        for symbol in code:
            code_word = []
            for bit in code[symbol]:
                if bit == "1":
                    code_word.append(HuffmanCoder.UINT8_1)
                elif bit == "0":
                    code_word.append(HuffmanCoder.UINT8_0)
                else:
                    raise RuntimeError
            code_numpy[symbol] = code_word

        encoded_list = []
        for byte in data:
            encoded_list += code_numpy[byte]

        padding = [HuffmanCoder.UINT8_0] * (8 - len(encoded_list) % 8)
        encoded_list += padding

        array = np.array(encoded_list, dtype=np.uint8)
        result = np.packbits(array)

        return result.tobytes(), len(padding)

    @staticmethod
    def _decode_data(data: bytes, tree: HuffmanTree, padding_length: int) -> str:
        """ Decodes binary data with Huffman coding

        :param data: Huffman encoded data
        :param tree: Huffman tree
        :return decoded UTF-8 string data
        """

        decoded_data = ""

        current_node = tree.root
        bits: np.array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        bits = bits[:-padding_length]

        bit_counter_processed = np.uint32(0)
        bit_counter = HuffmanCoder.UINT8_0
        for bit in bits:
            if bit == HuffmanCoder.UINT8_0 and current_node.left is not None:
                current_node = current_node.left
            elif bit == HuffmanCoder.UINT8_1 and current_node.right is not None:
                current_node = current_node.right

            if current_node.left is None and current_node.right is None:
                bit_counter_processed += bit_counter
                bit_counter = HuffmanCoder.UINT8_0
                decoded_data += current_node.symbol
                current_node = tree.root
            bit_counter += 1

        return decoded_data


def _main() -> None:
    """ Main function """

    parser = argparse.ArgumentParser()

    # Required argument
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output file")

    # Optional argument
    parser.add_argument("-d", "--decode", action="store_true",
                        help="Decode flag (with this flag program will work as decoder")
    parser.add_argument("-w", "--word-length", type=int, default=8, help="Input word length in bits")
    parser.add_argument("-b", "--block-size", type=int, default=1048576, help="Size of blocks used while encoding")

    args = parser.parse_args()

    if args.decode:
        HuffmanCoder.decode_file(args.input, args.output)
    else:
        HuffmanCoder.encode_file(args.input, args.output, args.block_size)


if __name__ == "__main__":
    _main()
