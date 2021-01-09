import numpy as np

UINT8_0 = np.uint8(0)
""" Constant value 0 of type uint8 """

UINT8_1 = np.uint8(1)
""" Constant value 1 of type uint8 """


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

    def encode(self, text_encoding: str = "utf-8"):
        """ Encodes Huffman tree to binary form (filled with 0 to full bytes)

        :param text_encoding: text encoding of tree symbols
        :return encoded Huffman tree
        """

        encoded = self._encode_node(self.root, text_encoding)
        padding = np.zeros(shape=(8 - len(encoded) % 8,), dtype=np.uint8)
        encoded = np.append(encoded, padding)
        return np.packbits(encoded).tobytes()

    @staticmethod
    def decode(buffer: bytes, text_encoding: str = "utf-8"):
        """ Decodes Huffman tree from binary form

        :param buffer: bytes to be decoded into huffman tree
        :param text_encoding: text encoding of tree symbols
        :return decoded Huffman tree
        """

        array: np.array = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))
        return HuffmanTree(HuffmanTree._decode_node(array, text_encoding, Counter()))

    @staticmethod
    def _encode_node(node: HuffmanNode, text_encoding: str,
                     bits: np.array = np.empty(shape=(0,), dtype=np.uint8)) -> np.array:
        """ Recursively encodes Huffman tree node

        :param node: Huffman tree node
        :param bits: bits of already encoded Huffman tree nodes
        :return encoded Huffman tree node
        """
        global UINT8_0, UINT8_1

        if node.left is None and node.right is None:
            bits = np.append(bits, UINT8_1)

            symbol_bits = np.unpackbits(np.frombuffer(node.symbol.encode(encoding=text_encoding), dtype=np.uint8))
            bits = np.append(bits, symbol_bits)
        else:
            bits = np.append(bits, UINT8_0)
            if node.left is not None:
                bits = HuffmanTree._encode_node(node.left, text_encoding, bits)
            if node.right is not None:
                bits = HuffmanTree._encode_node(node.right, text_encoding, bits)

        return bits

    @staticmethod
    def _decode_node(bits: np.array, text_encoding: str, counter: Counter) -> HuffmanNode:
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
            symbol = np.packbits(symbol_bits).tobytes().decode(encoding=text_encoding)
            return HuffmanNode(0, symbol=symbol)
        else:
            left_node = HuffmanTree._decode_node(bits, text_encoding, counter)
            right_node = HuffmanTree._decode_node(bits, text_encoding, counter)
            return HuffmanNode(0, left_node, right_node)
