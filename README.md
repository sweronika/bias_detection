
## Bias detection in Wikipedia articles
### A study on Polish and English datasets.

This repository embrace Polish equivalent of the [WNC dataset](https://github.com/rpryzant/neutralizing-bias) which was the first parallel corpus of biased language. 
Similarly, we introduce the Polish corpus containing examples of biased and unbiased sentences scraped from Wikipedia articles. 
Part of the data was harvested with the use of code published by the WNC authors.

Collected data consists of subsets of sentences with slightly different characteristics.
- **biased** - biased sentences tagged by Wikipedia reviewers
- **reviewed** - reviewed version of previosly biased sentences
- **unbiased** - sentences from reviewed articles that were not marked as biased
- **featured** - sentences downloaded from [featured articles](https://pl.wikipedia.org/wiki/Wikipedia:Artyku%C5%82y_na_Medal) - articles of a very high quality
- **deceiving sentences** - sentences tagges as [deceiving](https://pl.wikipedia.org/wiki/Kategoria:Artyku%C5%82y_z_wyra%C5%BCeniami_zwodniczymi)

Files:
- **dataPL** - Polish subsets in a slighlty preprocessed format (depends on a subset)
- **00_data_PL.ipynb** - preprocessing and analysis of the Polish data
- **00_data_PL_EN.ipynb** - preprocessing and analysis of both Polish and English data

