import argparse
import time
from typing import List, Dict

import numpy as np

from src.text_entropy.calculate_text_entropy import count_symbols_in_string


class Counter:
    def __init__(self, int_value: int):
        self.value_ = int_value

    def get(self):
        value_ = self.value_
        self.value_ += 1
        return value_

    def increase(self, inc_value_):
        self.value_ += inc_value_

    def __repr__(self):
        return str(self.value_)


class HuffmanNode:
    def __init__(self, priority: float, left=None, right=None, symbol: str = ""):
        self.priority: float = priority
        self.left = left
        self.right = right
        self.symbol: str = symbol

    def __eq__(self, other):
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority \
               or (self.priority == other.priority and self.symbol < other.symbol)

    def __repr__(self):
        return f'HuffmanNode(priority = {self.priority}, symbol = "{self.symbol}", ' \
               f'hasLeft = {self.left is not None}, hasRight = {self.right is not None})'


class HuffmanTree:
    def __init__(self, root: HuffmanNode):
        self.root = root

    def encode(self):
        encoded = self._encode_node(self.root)
        padding = np.zeros(shape=(8 - len(encoded) % 8,), dtype=np.uint8)
        encoded = np.append(encoded, padding)
        return np.packbits(encoded).tobytes()

    @staticmethod
    def decode(buffer: bytes):
        array: np.array = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))
        return HuffmanTree(HuffmanTree._decode_node(array, Counter(0)))

    @staticmethod
    def _encode_node(node: HuffmanNode, bits: np.array = np.empty(shape=(0,), dtype=np.uint8)) -> np.array:
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
    def _decode_node(array: np.array, counter: Counter) -> HuffmanNode:
        index = counter.get()
        bit = array[index]
        if bit == np.uint(1):
            symbol_bits: np.array = array[index + 1:index + 9]
            counter.increase(8)
            symbol = np.packbits(symbol_bits).tobytes().decode()
            return HuffmanNode(0, symbol=symbol)
        else:
            left_node = HuffmanTree._decode_node(array, counter)
            right_node = HuffmanTree._decode_node(array, counter)
            return HuffmanNode(0, left_node, right_node)


class HuffmanCoder:
    UINT8_0 = np.uint8(0)
    UINT8_1 = np.uint8(1)

    @staticmethod
    def encode_file(in_file_path: str, out_file_path: str):
        with open(out_file_path, mode="wb") as output_file, open(in_file_path, mode="r") as input_file:
            output_file.write(bytes())

            frequencies: Dict[str, int] = {}
            while True:
                data = input_file.read(1024 * 128)
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

            while True:
                data = input_file.read(1024 * 128)
                if not data:
                    break

                encoded, padding_length = HuffmanCoder._encode_data(data, tree)
                data_length_bytes = len(encoded).to_bytes(4, byteorder="big", signed=False)
                padding_length_bytes = padding_length.to_bytes(1, byteorder="big", signed=False)

                block = data_length_bytes + padding_length_bytes + encoded
                output_file.write(block)

    @staticmethod
    def decode_file(in_file_path: str, out_file_path: str):
        with open(in_file_path, mode="rb") as input_file, open(out_file_path, mode="w") as output_file:
            output_file.write("")

            tree_length = int.from_bytes(input_file.read(4), byteorder="big", signed=False)  # In bytes
            tree: HuffmanTree = HuffmanTree.decode(input_file.read(tree_length))

            while True:
                block_length = int.from_bytes(input_file.read(4), byteorder="big", signed=False)  # In bytes
                padding_length = int.from_bytes(input_file.read(1), byteorder="big", signed=False)  # In bits
                data_bytes = input_file.read(block_length)

                if data_bytes == bytes():
                    break

                decoded = HuffmanCoder._decode_data(data_bytes, tree, padding_length)
                output_file.write(decoded)

    @staticmethod
    def _create_tree(frequencies: Dict[str, float]) -> HuffmanTree:
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
    def _node_to_dict(node: HuffmanNode, prefix: str = "", code: dict = None):
        """ Recursively walk the tree down to the leaves, assigning a src value to each symbol
        """

        if code is None:
            code = {}

        if node.left is not None:
            HuffmanCoder._node_to_dict(node.left, prefix + "0", code)
        else:
            code[node.symbol] = prefix
        if node.right is not None:
            HuffmanCoder._node_to_dict(node.right, prefix + "1", code)
        else:
            code[node.symbol] = prefix
        return code

    @staticmethod
    def _tree_to_dict(tree: HuffmanTree):
        return HuffmanCoder._node_to_dict(tree.root)

    @staticmethod
    def _encode_data(data: str, tree: HuffmanTree):
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
    def _decode_data(text: bytes, tree: HuffmanTree, padding_length: int):
        decoded_text = ""

        current_node = tree.root
        bits = np.unpackbits(np.frombuffer(text, dtype=np.uint8))
        bits = bits[:-padding_length]

        bit_counter_processed = np.uint32(0)
        bit_counter = HuffmanCoder.UINT8_0
        for bit in bits:
            if bit == HuffmanCoder.UINT8_0 and current_node.left is not None:
                current_node = current_node.left
            elif bit == HuffmanCoder.UINT8_1 and current_node.right is not None:
                current_node = current_node.right

            if current_node.symbol == "\00":
                return decoded_text
            elif current_node.left is None and current_node.right is None:
                bit_counter_processed += bit_counter
                bit_counter = HuffmanCoder.UINT8_0
                decoded_text += current_node.symbol
                current_node = tree.root
            bit_counter += 1

        return decoded_text


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--word-length", type=int, default=8, help="Input word length in bits")
    args = parser.parse_args()

    time_start = time.time()
    HuffmanCoder.encode_file("test.txt", "test_out.hf")
    print(f"Encoding took : {round(time.time() - time_start, 3)}")

    time_start = time.time()
    HuffmanCoder.decode_file("test_out.hf", "test_out.txt")
    print(f"Decoding took : {round(time.time() - time_start, 3)}")


if __name__ == "__main__":
    _main()
