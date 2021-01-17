""" This script calculates entropy of given text from file (UTF-8 encoding) """

NON_PRINTABLE_SYMBOLS = {
    " ": "\\s",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t"
}
""" dictionary with printable form for non-printable symbols """


def count_symbols_in_string(text: str, escape_symbols: bool = False, output_file_path: str = "") -> dict:
    """ Counts symbols in text from file and saves result to dict and to CSV file

    :param text:
    :param escape_symbols: if true non printable symbols are replaced with printable representation
    :param output_file_path: path of output file, if set to "" file will not be written
    :return: dictionary containing symbols as keys and their frequency as values
    """

    csv = "symbol|count\n"
    symbol_dict: dict = {}
    for symbol in sorted(set(text)):
        printable_symbol = symbol

        if escape_symbols and symbol in NON_PRINTABLE_SYMBOLS:
            printable_symbol = NON_PRINTABLE_SYMBOLS[symbol]

        symbol_count = text.count(symbol)
        symbol_dict[printable_symbol] = symbol_count
        if output_file_path != "":
            print(f"{printable_symbol}|{symbol_count}")
        csv += f"{printable_symbol}|{symbol_count}\n"

    if output_file_path == "":
        return symbol_dict

    with open(f"{output_file_path.replace('.txt', '_symbols.csv')}", mode="w+", encoding="utf-8") as file:
        file.write(csv)

    return symbol_dict


def count_symbols_in_file(file_path: str) -> dict:
    """ Counts symbols in text from file and saves result to dict and to CSV file

    :param file_path: path to file which contains UTF-8 text
    :return: dictionary containing symbols as keys and their frequency as values
    """

    global NON_PRINTABLE_SYMBOLS

    with open(file_path, mode="r+", encoding="utf-8") as file:
        text: str = file.read()
    return count_symbols_in_string(text)


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
    symbol_dict: dict = count_symbols_in_file(args.file)
    entropy_csv = _calculate_entropy(symbol_dict)

    # Save result (entropy) to file
    with open(f"{args.file.replace('.txt', '_entropy.csv')}", mode="w+", encoding="utf-8") as file:
        file.write(entropy_csv)


if __name__ == "__main__":
    _main()
