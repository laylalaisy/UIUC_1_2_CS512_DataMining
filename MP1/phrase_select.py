import re

def main():
	input_filename = './phrase.emb'
	output_filename = './phrase_selected.emb'
	pattern = re.compile(r'_(.*?)_')
	with open(input_filename) as input:
		with open(output_filename, "w") as output:
			for line in input.readlines():
				value = line.split(' ')
				if re.match(pattern, line):
					output.write(line)


if __name__ == "__main__":
    main()
