library(tsne)
library(ggplot2)
args = commandArgs(trailingOnly=TRUE)
path <- args[1]
vocab <- readLines(file('data/vocab.txt', 'r'))
embeds_files <- Sys.glob(paste(path, '*embeddings_from*', sep='/'))
print(embeds_files)

print(vocab)
# TODO size of vocab <- size of embeds - 1 -- repair this

for (embeds_file in embeds_files){
	embeds <- read.csv(embeds_file, header=F)
	projection <- data.frame(tsne(embeds))
	colnames(projection) <- c('v1', 'v2')
	ggplot(projection, aes(x=v1, y=v2)) + geom_point()
	ggsave(paste(embeds_file, 'png', sep='.'))
}
