Table of contents
=================


   ##### [1. Presentation](#Presentation)
   ##### [2. Neural language understanding challenges](#)
   ##### [3. Functionnalities](#)
   ##### [4. The key idea behind ELMo](#)
   * ###### [4.1. Traditional techniques](#)
   * ###### [4.2. Previous work](#)
   * ###### [4.3. Technical details](#)
      * ###### [4.3.1.  Forward](#)
      * ###### [4.3.2.  Backward](#)
      * ###### [4.3.3.  BiLM](#)
   ##### [5. Using BiLM for supervised learning](#Features)



### 1. Presentation:
The paper introduces a new type of deep contextualized word representation that directly addresses both challenges, can be easily integrated into existing models, and significantly improves the state of the art in every considered case across a range of challenging language understanding problems.
 The word vectors are learned functions of the internal states of a deep bidirectional language model __(biLM)__, which is pretrained on a large text corpus. Its shown that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis.
 
 ### 2. Neural language understanding challenges:
Pre-trained word representations __(Mikolov et al., 2013; Pennington et al., 2014)__ are a key component in many neural language understanding models. However, learning high quality representations can be challenging.
 
### 3. Functionnalities:
Deep contextualized word representation models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). 

### 4. The key idea behind ELMo:
It use vectors derived from a bidirectional LSTM that is trained with a coupled language model __(LM)__ objective on a large text corpus. For this reason, it are called __ELMo__ (Embeddings from Language Models) representations.
ELMo representations are deep, in the sense that they are a function of all of the internal layers of the biLM. More specifically, it learns a linear combination of the vectors stacked above each input word for each end task, which markedly improves performance over just using the top LSTM layer.
Combining the internal states in this manner allows for very rich word representations. Using intrinsic evaluations, we show that the higher-level LSTM states capture context-dependent aspects of word meaning (e.g., they can be used without modification to perform well on supervised word sense disambiguation tasks) while lowerlevel states model aspects of syntax (e.g., they can be used to do part-of-speech tagging). Simultaneously exposing all of these signals is highly beneficial, allowing the learned models select the types of semi-supervision that are most useful for each end task.


### 4.1. Traditional techniques:
Traditional word type embeddings in that each token is assigned a representation that is a function of the entire input sentence. 
### 4.2. Previous work:
Traditional approaches for learning word vectors only allow a single context independent representation for each word.
- Traditional word vectors by either enriching them with subword information __(e.g., Wieting et al., 2016; Bojanowski et al., 2017)__ or learning separate vectors for each word sense __(e.g., Neelakantan et al., 2014)__.
- Other works focused on learning context-dependent representations. context2vec __(Melamud et al., 2016)__ uses a bidirectional Long Short Term Memory __(LSTM; Hochreiter and Schmidhuber, 1997)__ to encode the context around a pivot word. 
- Other approaches for learning contextual embeddings include the pivot word itself in the representation and are computed with the encoder of either a supervised neural machine translation __(MT)__ system __(CoVe; McCann et al., 2017)__ or an unsupervised language model __(Peters et al., 2017)__.

### 4.3. Technical details:
ELMo word representations are functions of the entire input sentence, as described in this section. They are computed on top of two-layer biLMs with character convolutions, as a linear function of the internal network states. This setup allows to do semi-supervised learning, where the biLM is pretrained at a large scale and easily incorporated into a wide range of existing neural NLP architectures.
##### 4.3.1.  Forward:
Neural language models __(Jozefowicz et al. ´ , 2016; Melis et al., 2017; Merity et al., 2017)__ compute a context-independent token representation (via token embeddings or a CNN over characters) then pass it through L layers of forward LSTMs. At each position k, each LSTM layer outputs a context-dependent representation. The top layer LSTM output, is used to predict the next token with a Softmax layer
[!image](https://github.com/oussama-benbrahem/Natural-Questions-Competition-Google-ai/blob/master/images/ELMo_forward.
##### 4.3.2. Backward:
Backward LM is similar to a forward LM, except it runs over the sequence in reverse, predicting the previous token given the future context:

[!image](https://github.com/oussama-benbrahem/Natural-Questions-Competition-Google-ai/blob/master/images/ELMo_backward.png)

##### 4.3.3. BiLM:
A biLM combines both a forward and backward LM. Our formulation jointly maximizes the log likelihood of the forward and backward directions:
[!image](https://github.com/oussama-benbrahem/Natural-Questions-Competition-Google-ai/blob/master/images/ELMo_biLM.png)
Parameters for both the token representation (Θx) and Softmax layer (Θs) in the forward and backward direction while maintaining separate parameters for the LSTMs in each direction. Overall, this formulation is similar to the approach of __Peters et al. (2017)__, with the exception that it shares some weights between directions instead of using completely independent parameters.

### 5. Using BiLM for supervised learning:

BiLM is used in various supervised learning and the model is presented as follow:
First consider the lowest layers of the supervised model without the biLM. Most supervised NLP models share a common architecture at the lowest layers, allowing us to add ELMo in a consistent, unified manner. Given a sequence of tokens (t1, . . . , tN), it is standard to form a context-independent token representation x_k for each token position using pre-trained word embeddings and optionally character-based representations. Then, the model forms a context-sensitive representation h_k, typically using either bidirectional RNNs, CNNs, or feed forward networks.
you can add ELMo to your supervised model, by firstly freezing the weights of the biLM and then concatenate the ELMo vector ELMotask (k) with x_k and pass the ELMo enhanced representation into the task RNN. For some tasks (e.g., SNLI, SQuAD), we observe further improvements by also including ELMo at the output of the task RNN by introducing another set of output specific linear weights and replacing h_k. As the remainder of the supervised model remains unchanged, these additions can happen within the context of more complex neural models. For example, see the SNLI experiments in Sec. 4 where a bi-attention layer$$ follows the biLSTMs, or the coreference resolution experiments where a clustering model is layered on top of the biLSTMs. Finally, we found it beneficial to add a moderate amount of dropout to ELMo __(Srivastava et al., 2014)__ and in some cases to regularize the ELMo weights by adding λkwk to the loss. This imposes an inductive bias on the ELMo weights to stay close to an average of all biLM layers.

 




