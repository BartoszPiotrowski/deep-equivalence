library(ggplot2)

VOCAB_FILE<-'data/vocab.txt'
EMBED_FILE_PATTERN<-'*embeddings*.csv'
PADDING_SYMBOL<-'<pad>'

args <- commandArgs(trailingOnly=TRUE)
path <- args[1]
vocab <- readLines(file(VOCAB_FILE, 'r'))
symbols <- c(PADDING_SYMBOL, vocab)
embeds_files <- Sys.glob(paste(path, EMBED_FILE_PATTERN, sep='/'))

for (embeds_file in embeds_files){
	embeds <- read.csv(embeds_file, header=F)
	colnames(embeds) <- c('v1', 'v2')
	embeds['symbol'] <- symbols
	# TODO fixed grid
	ggplot(embeds, aes(x=v1, y=v2, label=symbol)) +
	       	geom_text(aes(label=symbol)) +
		scale_x_continuous(limits = c(-3, 3)) +
		scale_y_continuous(limits = c(-3, 3))
	ggsave(sub('.csv', '.png', embeds_file))
}
