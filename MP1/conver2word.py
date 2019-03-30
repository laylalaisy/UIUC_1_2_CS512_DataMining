import re

file = open('./AutoPhrase/models/nyt/segmentation.txt')
contents = file.read()

for content in re.findall('<phrase>[\s\S]*?</phrase>', contents):
	content_new = re.sub(' ', '_', content)
	content_new = re.sub('<phrase>', '_', content_new)
	content_new = re.sub('</phrase>', '_', content_new)
	# print(content_new)
	contents = re.sub(content, content_new, contents)

file_new = open('./AutoPhrase/models/nyt/nyt_segmentation.txt', 'w')
file_new.write(contents)
file_new.close()