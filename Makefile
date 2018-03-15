
.PHONY: zipper, unzipper
zipper: data
	zip -r the_data.zip data
unzipper: the_data.zip
	unzip the_data.zip
