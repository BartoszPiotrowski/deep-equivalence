library(ggplot2)

VOCAB_FILE<-'data/vocab.txt'
EMBED_FILE_PATTERN<-'*embeddings*.csv'
PADDING_SYMBOL<-''

args <- commandArgs(trailingOnly=TRUE)
path <- args[1]
vocab <- readLines(file(VOCAB_FILE, 'r'))
symbols <- c(PADDING_SYMBOL, vocab)
embeds_files <- Sys.glob(paste(path, EMBED_FILE_PATTERN, sep='/'))

for (embeds_file in embeds_files){
	embeds <- read.csv(embeds_file, header=F)
	pca <- prcomp(embeds)
	projection <- data.frame(predict(pca, embeds)[,c(1,2)])
	projection['symbol'] <- symbols
	ggplot(projection, aes(x=PC1, y=PC2, label=symbol)) +
	       	geom_text(aes(label=symbol)) +
		scale_x_continuous(limits = c(-3, 3)) +
		scale_y_continuous(limits = c(-3, 3))

	ggsave(sub('.csv', '.png', embeds_file))
}
