library(tsne)
library(ggplot2)

VOCAB_FILE<-'data/vocab.txt'
EMBED_FILE_PATTERN<-'*embeddings_from*.csv'
PADDING_SYMBOL<-'<pad>'

args <- commandArgs(trailingOnly=TRUE)
path <- args[1]
vocab <- readLines(file(VOCAB_FILE, 'r'))
symbols <- c(PADDING_SYMBOL, vocab)
embeds_files <- Sys.glob(paste(path, EMBED_FILE_PATTERN, sep='/'))

for (embeds_file in embeds_files){
	print(embeds_file)
	embeds <- read.csv(embeds_file, header=F)
	projection <- data.frame(tsne(embeds))
	colnames(projection) <- c('v1', 'v2')
	projection['symbol'] <- symbols
	# TODO fixed grid
	ggplot(projection, aes(x=v1, y=v2, label=symbol)) +
	       	geom_text(aes(label=symbol))
	print(embeds_file)
	ggsave(sub('.csv', '.png', embeds_file))
}
