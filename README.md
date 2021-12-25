# NLP Language Model from Scratch

In this project, our aim is to create character n-gram language models for the different languages. A character language model will typically assign higher probabilities (and so, lower perplexities) to text in the same language as the text it was trained on. We will use this property of language models for language identification.

## How to Execute?

To run this project,

1. Download the repository as a zip file.
2. Extract the zip to get the project folder.
3. Open Terminal in the directory you extracted the project folder to. 
4. Change directory to the project folder using:

    `cd language-model-from-scratch-main`
5. Install the required libraries, **NLTK** and **NumPy** using the following commands:

    `pip3 install nltk`

    `pip3 install numpy`
 
6. Now to execute the code, use any of the following commands (in the current directory):

**Unsmoothed model:**
`python3 src/main.py data/train/ data/dev/ output/results_dev_unsmoothed.csv --unsmoothed`

**Laplace Smoothed model:**
`python3 src/main.py data/train/ data/dev/ output/results_dev_laplace.csv --laplace`

**Linear Interpolation model:**
`python3 src/main.py data/train/ data/dev/ output/results_dev_interpolation.csv --interpolation`
    
    
## Description of the execution command


Our program **src/main.py** that takes three positional command-line arguments and an optional command-line argument. The three positional arguments should be in this order: the first is a path to the training data folder, the second is a path to the test data folder, and the third is a path to the output csv file. The optional argument should be one of the three to select the type of model to train: "--unsmoothed", "--laplace", or "--interpolation". When there is no optional argument specified, the program will default to the unsmoothed model.

The assignment's training data is at [data/train](data/train) and the development data is at [data/dev](data/dev). That's why we have specified these paths in the execution commands above.


## References

https://coderedirect.com/questions/344874/nltk-package-to-estimate-the-unigram-perplexity

https://dev.to/amananandrai/language-model-implementation-bigram-model-22ij

https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity

https://stathwang.github.io/part-of-speech-tagging-with-trigram-hidden-markov-models-and-the-viterbi-algorithm.html

https://medium.com/mti-technology/n-gram-language-model-b7c2fc322799

https://towardsdatascience.com/perplexity-in-language-models-87a196019a94

https://github.com/mmera/ngram-model/blob/f37004e8d14d7d595787621b83886807bd6510e5/ngram.py#L61

https://stackoverflow.com/questions/1217251/sorting-a-dictionary-with-lists-as-values-according-to-an-element-from-the-list
                           
