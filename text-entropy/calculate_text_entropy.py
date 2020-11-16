""" This script calculates entropy of given text from file (UTF-8 encoding) """

NON_PRINTABLE_SYMBOLS = {
    " ": "\\s",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t"
}
""" dictionary with printable form for non-printable symbols """


def _count_symbols_in_text(filename: str) -> dict:
    """ Counts symbols in text from file and saves result to dict and to CSV file

    :param filename
    :return: dictionary containing symbols as keys and their frequency as values
    """

    global NON_PRINTABLE_SYMBOLS

    with open(filename, mode="r+", encoding="utf-8") as file:
        text: str = file.read()

    csv = "symbol|count\n"
    symbol_dict: dict = {}
    for symbol in sorted(set(text)):
        printable_symbol = symbol

        if symbol in NON_PRINTABLE_SYMBOLS:
            printable_symbol = NON_PRINTABLE_SYMBOLS[symbol]

        symbol_count = text.count(symbol)
        symbol_dict[printable_symbol] = symbol_count
        print(f"{printable_symbol}|{symbol_count}")
        csv += f"{printable_symbol}|{symbol_count}\n"

    with open(f"{filename.replace('.txt', '_symbols.csv')}", mode="w+", encoding="utf-8") as file:
        file.write(csv)

    return symbol_dict


def _calculate_entropy(symbol_dict: dict) -> str:
    """ Calculates entropy

    :param symbol_dict: dictionary consisting of symbols as keys and their frequency as values
    :return: CSV as str containing entropies of symbols and whole text
    """

    import math

    total_symbol_count = sum(symbol_dict.values())

    entropy = 0
    csv = "symbol|information\n"
    for symbol in symbol_dict:
        probability = symbol_dict[symbol] / total_symbol_count
        symbol_information = math.log(1 / probability, 2)

        csv += f"{symbol}|{symbol_information}\n"
        entropy += probability * symbol_information

    csv += "|\n"
    csv += f"ENTROPY|{entropy}\n"

    print()
    print(f"Text entropy : {entropy}")
    return csv


def _main() -> None:
    """ Main function """

    # Get console arguments
    import argparse
    parser = argparse.ArgumentParser(description="Count symbols in text.")
    parser.add_argument("-f", "--file", type=str, help="path to file", required=True)
    args = parser.parse_args()

    # Get result
    symbol_dict: dict = _count_symbols_in_text(args.file)
    entropy_csv = _calculate_entropy(symbol_dict)

    # Save result (entropy) to file
    with open(f"{args.file.replace('.txt', '_entropy.csv')}", mode="w+", encoding="utf-8") as file:
        file.write(entropy_csv)


if __name__ == "__main__":
    _main()
