""" This script handles encoding and decoding of UTF-8 files with Huffman coding """

import argparse
import multiprocessing
import os
import time
from multiprocessing.pool import Pool, ApplyResult
from typing import List, Dict, Tuple

import numpy as np
from console_progressbar import ProgressBar

from src.structures import HuffmanTree, HuffmanNode, UINT8_1, UINT8_0
from src.text_entropy.calculate_text_entropy import count_symbols_in_string


class HuffmanCodec:
    """ This class provides Huffman encoding and decoding functionality """

    @staticmethod
    def encode_file(in_file_path: str, out_file_path: str, text_encoding: str = "utf-8",
                    block_size: int = 1024 * 1024, verbose: bool = False) -> None:
        """ Encodes file with Huffman coding

        :param in_file_path: path to input file
        :param out_file_path: path to output file
        :param block_size: size of blocks to be used in encoding
        :param text_encoding: text encoding of input file
        :param verbose: if set to True more information is printed
        """

        with open(in_file_path, mode="r", encoding=text_encoding) as input_file, \
                open(out_file_path, mode="wb") as output_file:
            output_file.write(bytes())

            # First count of each character in text are obtained
            symbol_counts: Dict[str, int] = {}
            while True:
                data = input_file.read(block_size)
                if not data:
                    break

                frequencies_part = count_symbols_in_string(data)
                for key in frequencies_part:
                    if key in symbol_counts:
                        symbol_counts[key] = symbol_counts[key] + frequencies_part[key]
                    else:
                        symbol_counts[key] = frequencies_part[key]
            # Moves input file pointer to start position
            input_file.seek(0)

            if verbose:
                print("Symbol counts obtained")
                print(str(symbol_counts).replace("{", "").replace("}", "").replace(", ", "\n"))
                print()

            # Secondly text encoding is saved to output file
            text_encoding_length = len(text_encoding).to_bytes(1, byteorder="big", signed=False)
            output_file.write(text_encoding_length)
            output_file.write(text_encoding.encode(encoding="ascii"))

            if verbose:
                print("Text encoding saved")
                print(f"Text encoding length : {text_encoding_length}")
                print(f"Text encoding : {text_encoding}")
                print()

            # Thirdly tree is created, encoded and saved to output file
            tree: HuffmanTree = HuffmanCodec._create_tree(symbol_counts)

            encoded_tree = tree.encode(text_encoding=text_encoding)
            encoded_tree_length_bytes = len(encoded_tree).to_bytes(4, byteorder="big", signed=False)

            output_file.write(encoded_tree_length_bytes)
            output_file.write(encoded_tree)

            if verbose:
                print("Huffman tree created")
                print("Code")

                huffman_code: dict = HuffmanCodec._tree_to_dict(tree)
                print(str(huffman_code).replace("{", "").replace("}", "").replace(", ", "\n"))

                average_code_length = sum([len(code) for code in huffman_code.values()]) / len(huffman_code)

                print(f"Average code length [bits] : {round(average_code_length, 2)}")
                print()

            # Here progressbar is set upped
            file_stats = os.stat(in_file_path)
            data_size = file_stats.st_size
            pb = ProgressBar(total=data_size, prefix='Progress', suffix='finished', decimals=1, length=50, fill='X',
                             zfill='-')
            data_processed = 0

            # Lastly file content is encoded and saved to output file
            while True:
                pb.print_progress_bar(data_processed)
                data = input_file.read(block_size)
                if not data:
                    break

                data_processed += len(data)

                encoded, padding_length = HuffmanCodec._encode_data(data, tree)
                data_length_bytes = len(encoded).to_bytes(4, byteorder="big", signed=False)
                padding_length_bytes = padding_length.to_bytes(1, byteorder="big", signed=False)

                block = data_length_bytes + padding_length_bytes + encoded
                output_file.write(block)

            pb.print_progress_bar(data_size)

    @staticmethod
    def decode_file(in_file_path: str, out_file_path: str, verbose: bool = False) -> None:
        """ Encodes file with Huffman coding

        :param in_file_path: path to input file
        :param out_file_path: path to output file
        :param verbose: if set to True more information is printed
        """

        with open(in_file_path, mode="rb") as input_file:
            # First Huffman tree is decoded
            text_encoding_length = int.from_bytes(input_file.read(1), byteorder="big", signed=False)  # In bytes

            if verbose:
                print("Text encoding obtained")
                print(f"Text encoding length : {text_encoding_length}")

            text_encoding: str = input_file.read(text_encoding_length).decode("ascii")

            if verbose:
                print(f"Text encoding : {text_encoding}")
                print()

            # Secondly Huffman tree is decoded
            tree_length = int.from_bytes(input_file.read(4), byteorder="big", signed=False)  # In bytes
            tree: HuffmanTree = HuffmanTree.decode(input_file.read(tree_length), text_encoding)

            if verbose:
                print("Huffman tree decoded")
                print("Code")
                print(str(HuffmanCodec._tree_to_dict(tree)).replace("{", "").replace("}", "").replace(", ", "\n"))
                print()

            # Here progressbar is set upped
            file_stats = os.stat(in_file_path)
            data_size = file_stats.st_size
            pb = ProgressBar(total=data_size, prefix='Progress', suffix='finished',
                             decimals=1, length=50, fill='X', zfill='-')
            data_processed = 4 + tree_length

            # This pool will be used for data blocks decoding
            pool_jobs: List[Tuple[ApplyResult, int]] = []

            with open(out_file_path, mode="w", encoding=text_encoding) as output_file, \
                    Pool(processes=multiprocessing.cpu_count()) as pool:
                output_file.write("")

                # Thirdly block from file are read and decoded in separate process
                # Finish is set to True when all data is read
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
                            (pool.apply_async(HuffmanCodec._decode_data, (data_bytes, tree, padding_length)),
                             data_length)
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
        """ Create Huffman tree  from symbol frequencies

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
        """ Converts Huffman tree node to dictionary

        :param node: Huffman tree node
        :param prefix: bits that had been already red
        :param code_dict: incomplete code dictionary
        :return Huffman tree transformed to dictionary (key - symbol, value - Huffman code)
        """

        if code_dict is None:
            code_dict = {}

        if node.left is not None:
            HuffmanCodec._node_to_dict(node.left, prefix + "0", code_dict)
        else:
            code_dict[node.symbol] = prefix

        if node.right is not None:
            HuffmanCodec._node_to_dict(node.right, prefix + "1", code_dict)
        else:
            code_dict[node.symbol] = prefix

        return code_dict

    @staticmethod
    def _tree_to_dict(tree: HuffmanTree):
        """ Converts Huffman tree to dictionary

        :param tree: Huffman tree
        :return Huffman tree transformed to dictionary (key - symbol, value - Huffman code)
        """

        return HuffmanCodec._node_to_dict(tree.root)

    @staticmethod
    def _encode_data(data: str, tree: HuffmanTree) -> Tuple[bytes, int]:
        """ Encodes string data with Huffman coding

        :param data: UTF-8 string data
        :param tree: Huffman tree
        :return encoded data and length of padding (in bits)
        """

        code = HuffmanCodec._tree_to_dict(tree)
        code_numpy = {}

        for symbol in code:
            code_word = []
            for bit in code[symbol]:
                if bit == "1":
                    code_word.append(UINT8_1)
                elif bit == "0":
                    code_word.append(UINT8_0)
                else:
                    raise RuntimeError
            code_numpy[symbol] = code_word

        encoded_list = []
        for byte in data:
            encoded_list += code_numpy[byte]

        padding = [UINT8_0] * (8 - len(encoded_list) % 8)
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
        bit_counter = UINT8_0
        for bit in bits:
            if bit == UINT8_0 and current_node.left is not None:
                current_node = current_node.left
            elif bit == UINT8_1 and current_node.right is not None:
                current_node = current_node.right

            if current_node.left is None and current_node.right is None:
                bit_counter_processed += bit_counter
                bit_counter = UINT8_0
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
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose flag (prints more information)")
    parser.add_argument("-d", "--decode", action="store_true",
                        help="Decode flag (with this flag program will work as decoder")
    parser.add_argument("-te", "--text-encoding",
                        help="Text encoding of input text (look at: https://docs.python.org/3/library/codecs.html)",
                        default="utf-8")
    parser.add_argument("-b", "--block-size", type=int, default=1048576,
                        help="Size of blocks (in bytes) used while encoding")

    args = parser.parse_args()

    time_start = time.time()
    if args.decode:
        HuffmanCodec.decode_file(args.input, args.output, args.verbose)
    else:
        HuffmanCodec.encode_file(args.input, args.output, args.text_encoding, args.block_size, args.verbose)

    if args.verbose:
        print()
        print(f"Took : {round(time.time() - time_start, 3)}s")


if __name__ == "__main__":
    _main()
