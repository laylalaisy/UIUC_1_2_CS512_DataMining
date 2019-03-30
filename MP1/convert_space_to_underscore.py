import sys
import re


def convert_line(line, pattern):
    phrases = re.findall(pattern, line)
    for phrase in phrases:
        s = '_'
        s = ' _' + s.join(phrase.split(" ")) + '_ '
        line = re.sub(pattern, s, line, 1)
    return line


def main():
    input_filename = sys.argv[1]
    output_filename = input_filename + ".replace.txt"
    with open(input_filename) as input:
        with open(output_filename, "w") as output:
            output = open(output_filename, "w")
            pattern = re.compile(r'<phrase>(.*?)</phrase>')
            for line in input.readlines():
                output.write(convert_line(line, pattern))


if __name__ == "__main__":
    main()
