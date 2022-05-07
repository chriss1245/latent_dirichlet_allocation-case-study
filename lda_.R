setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data <- read.csv("tweet_emotions.csv")

# a quick view
unique(data$sentiment)
head(data$content)


#----------------get a sample--------------------------------------------------
indices <- data$sentiment == "sadness" | data$sentiment == "love" | data$sentiment == "hate"
data <- data[indices,1:2]

# Quick review
#proportion of sadness
mean(data$sentiment=="sadness")

#proportion of love
mean(data$sentiment=="love")

#proportion of hate
mean(data$sentiment=="hate")

# Sampling 1000 tweets per emotion
sad <- sample(which(data$sentiment=="sadness"), size=700, replace=F)
love <- sample(which(data$sentiment=="love"), size=700, replace=F)
hate <- sample(which(data$sentiment=="hate"), size=700, replace=F)

df <-data[c(sad, love, hate),]


#------------------Visuzalization and preprocessing--------------------------

library(tm)

# NLP
corpus <-  Corpus(VectorSource(df$content))
clean_corpus <-  tm_map(corpus, tolower)
clean_corpus <- tm_map(clean_corpus, removeNumbers)
clean_corpus <- tm_map(clean_corpus, removePunctuation)
clean_corpus <- tm_map(clean_corpus, removeWords,
                       stopwords("en"))

clean_corpus <- tm_map(clean_corpus, stripWhitespace)

# Visuzlization
library(wordcloud)
wordcloud(clean_corpus[1:700], max.words = 50)
wordcloud(clean_corpus[700:1400], max.words = 50)
wordcloud(clean_corpus[1400:2100], max.words = 50)


# Computing bag of words
bow = apply(TermDocumentMatrix(clean_corpus), 1, sum)


#-----------------Hyper parameters-----------------------------------------
# Number of topics K
topics <- 3

# Number of words N
words <- length(bow)

# Convert the corpus on a list of documents (vectors with words)
cc <- as.list(clean_corpus, function(x) strsplit(x, " "))

# Since the tweets are small. There is a lot of chance of getting a tweet which ends up empty after filtering stopwords.
non_empty = cc!=""
cc <- cc[cc!=""]

#Number of documents M
documents <- length(cc)

# Prior distributions
alpha <- c(1,1,1)
beta <- rep(1, words)
names(beta) = names(bow)

# Gibbs sampling hyperparameters
iterations <- 5000
burning <- 200
thining <- 5

#--------------Random initialization----------------------------------------
# List with words per document
doc_words = NULL

# List of topics each word belongs per document
topic_words = NULL

for (i in 1:(documents)){
  l <- list()
  words_ <- strsplit(cc[[i]], " ")[[1]]
  l[[1]]  <- words_[! is.na(bow[words_])]
  doc_words <- rbind(doc_words, l)
  
  l <- list()
  l[[1]] <- sample(seq(1:topics), replace=T, size=length(words_[[1]]))
  topic_words <- rbind(topic_words, l)
  
}


# Word to id converter
w2id <- seq(1,words)
names(w2id) <- names(bow)

### Computing count matrices
# Word per topic matrix
wpt <- matrix(0, nrow=words, ncol=topics)
rownames(wpt) <- names(bow)

# Topic per document matrix
tpd <- matrix(0, nrow=documents, ncol=topics)

m <- 1
for (doc_ in doc_words){
  n <- 1
  for (word_ in doc_){
    wi <- w2id[word_]
    t <- topic_words[[m]][n]
    wpt[wi, t]  = wpt[wi, t]  + 1
    tpd[m, t] = tpd[m, t] + 1
    
    n = n + 1
  }
  m = m + 1
}


#----------------Gibbs sampling------------------------------------------------
phis <- array(0, c(iterations, words, topics))
thetas <- array(0, c(iterations, documents, topics))
aux <- 1
for (iteration in 1:(iterations + burning)){
  m <- 1
  for (doc_ in doc_words){
    n <- 1
    for (word_ in doc_){
      wi <- w2id[word_]
      t <- topic_words[[m]][n]
      wpt[wi, t]  = wpt[wi, t]  - 1
      tpd[m, t] = tpd[m, t] - 1
      
      probs <- rep(0, topics)
      for (k in 1:topics){
        # multiply times 100 for stability
        word_odds <- (wpt[wi, k] + beta[wi])/(sum(beta) + sum(wpt[,k]) )
        topic_odds <- (tpd[m, k] + alpha[k])/(sum(alpha) + sum(tpd[,k])) 
        probs[k] <- topic_odds*word_odds
      }
      
      # Samplig
      new_t <- sample(1:topics, size=1, prob = probs/sum(probs))
      
      # Change the topic the word belongs to
      topic_words[[m]][n] = new_t
      
      # Updating counts
      wpt[wi, new_t]  = wpt[wi, new_t]  + 1
      tpd[m, new_t] = tpd[m, new_t] + 1
      n = n + 1
    }
    m = m + 1
  }
  
  # Saving iterations
  if(iteration > burning && iteration %% thining == 0){
    phi_ <- wpt+beta
    sum_phi <- apply(phi_, 2, sum)
    # normalizing the probabilities
    phi_ <- t(t(phi_)/sum_phi)
    phis[aux,,] <- phi_
    
    theta_ <- t(t(tpd) + alpha)
    sum_theta <- apply(theta_, 1, sum)
    theta_ <- theta_/sum_theta
    
    thetas[aux,,] <- theta_
    
    aux <- aux + 1
    if ( iteration %% 500 == 0)
      print("working")
  }
}

#-------------------------------Saving results----------------------------------
l <- list(phis=phis[1:aux,,], thetas=thetas[1:aux,,], wpt=wpt, tpd=tpd, cc=cc, 
     topic_words=topic_words, doc_words=doc_words, w2id=w2id, df=df[non_empty,],
     words=words, documents=documents, topics=topics)

save(l, file="results.RData")
