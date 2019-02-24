## Presentation:
The Biderctional Encoder Respresentations from Transformers (BERT) is a langugae representation model

1. Objective
BERT is desingned to  pre-train deep bidirectional representations by jointly conditionning on both left and right context in all layers.

#### Existiong techniques
There are two existing startegies for applying pre-trained language representations to downstream 
### Feature-based approach:
uses tasks-specific architectures that include the pre-trained architrectures as additional features such as ELMo (Peters et al., 2018)
### Fine-tuning approach:
The fine-tuning approach introduces minimal task-specific parameters, and is trained on the down stream tasks by simply fine-tuning the pre-trained parameters. For example Generative pre-trained Transformer (Radford et al., 2018)
### Limitations
-  restrict the power of the pre-trained representations, especially for the fine-tuning approaches. 
-  standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training
-   incorporate context from both directions.
### The key idea behind BERT:
improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers.
BERT addresses the previously mentioned unidirectional constraints by proposing a new pre-training objective: the “masked language model” (MLM), inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective allows the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also introduce a “next sentence prediction” task that jointly pre-trains text-pair representations.

### BERT contributions:
- demonstrate the importance of bidirectional pre-training for language representations
- it shows that pre-trained representations eliminate the needs of many heavily engineered task-specific architectures
- BERT advances the state-of-the-art for eleven NLP tasks

# Relevant works:
#### feature-based approaches:
ELMo (Peters et al., 2017) generalizes traditional word embedding research along a different dimension. They propose to extract contextsensitive features from a language model. When integrating contextual word embeddings with existing task-specific architectures
#### fine-tuning approaches:
A recent trend in transfer learning from language models (LMs) is to pre-train some model architecture on a LM objective before fine-tuning that same model for a supervised downstream task (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018). The advantage of these approaches is that few parameters need to be learned from scratch.

![image]
![image](https://github.com/oussama-benbrahem/Natural-Questions-Competition-Google-ai/blob/master/images/bert.png)
![GitHub Logo](https://github.com/oussama-benbrahem/Natural-Questions-Competition-Google-ai/blob/master/images/bert.png)
### Input Representation
• The first token of every sequence is always the special classification embedding
([CLS]). The final hidden state (i.e., output of Transformer) corresponding to this token is used as the aggregate sequence representation for classification tasks. For nonclassification tasks, this vector is ignored.
• Sentence pairs are packed together into a single sequence. We differentiate the sentences
in two ways. First, we separate them with
a special token ([SEP]). Second, we add a
learned sentence A embedding to every token
of the first sentence and a sentence B embedding to every token of the second sentence.
• For single-sentence inputs we only use the
sentence A embeddings.

#### Pre-training Tasks
* Masked LM:
standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself” in a multi-layered context. In order to train a deep bidirectional representation, we take a straightforward approach of masking some percentage of the input tokens at random, and then predicting only those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953). In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.
    * example: , e.g., in the sentence my dog is hairy it chooses hairy. It then performs the following procedure:
    Rather than always replacing the chosen words with [MASK], the data generator will do the following:
        • 80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]
    • 10% of the time: Replace the word with a random word, e.g., my dog is hairy → my dog is apple
    • 10% of the time: Keep the word unchanged, e.g., my dog is hairy → my dog is hairy. The purpose of this is to bias the representation towards the actual observed word.
The Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Additionally, because random replacement only occurs for 1.5% of all tokens (i.e., 10% of 15%), this does not seem to harm the model’s language understanding capability.

* Next Sentence Prediction
In order to train a model that understands sentence relationships, we pre-train a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically,
when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A, and 50% of the time it is a random sentence from the corpus. For example: 
- Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
--> Label = IsNext
- Input = [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
--> Label = NotNext

### Implementations:
- Tensorflow: https://github.com/google-research/bert
- Keras: https://github.com/Separius/BERT-keras

