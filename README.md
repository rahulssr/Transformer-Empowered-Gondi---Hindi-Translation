# Weed-Pro :  Precision Weed Detection and Segmentation 🌱🌱 

Languages transcend mere communication; they embody living expressions of culture, history, and identity. In today's swiftly evolving world, numerous languages hover on the brink of extinction, risking the loss of invaluable cultural narratives and shared histories. The urgency to preserve indigenous dialects is palpable, and among them stands Gondi. The Gondi people, self-referenced as "Koitur," intricately weave into India's cultural tapestry as native speakers of the Dravidian language, Gondi. Their presence spans states like Madhya Pradesh, Maharashtra, Chhattisgarh, and beyond. 🌍

Despite their historical footprint, marked by kingdoms like Gondwana and the Garha Kingdom, the Gondi language, with its ties to Telugu, grapples with modernization challenges and the looming shadow of larger regional languages such as Hindi and Marathi. As of the 2001 census, the Gond population stood at approximately 11 million. However, linguistic assimilation, coupled with sociopolitical challenges like the Naxalite–Maoist insurgency, intensifies the need to safeguard and promote their distinctive linguistic heritage. 🗣️🛡️

Our endeavor addresses the imminent threat of language extinction, specifically focusing on Hindi-Gondi translation. The goal is to facilitate social upliftment by enabling seamless communication between these diverse linguistic groups. Our proposed methodology employs a Transformer-based Neural Machine Translation (NMT) model, initiated by a rigorous pre-processing pipeline for data sanitization and dialect identification. This sets the stage for accurate translations across various Gondi dialects, including Dorla, Koya, Madiya, Muria, or Raj Gond. 🚀

The subsequent stage involves implementing a Transformer-based NMT model enriched with innovative features such as weighted multi-head attention and learnable positional encoding. These elements collaborate harmoniously to process text inputs effectively. The model's quality and efficacy are showcased through a high-performance BLEU score, reinforcing its potential as a tool to bridge the communication divide among these diverse language groups. 📊✨

<img src = Conceptuall-1.png  width = "550" height = "400">


## Dataset Collection 🔍🔍


Alright, let's give it another shot, incorporating emojis for visual emphasis:


The Gondi-speaking community, spanning six Indian states, has undergone significant dialectical variations due to the influence of the dominant languages in each region. This linguistic diversity, while a testament to cultural interplay, has occasionally hindered clear communication among Gond speakers from different areas. 🌐 To address this challenge, we organized a series of comprehensive workshops with Gondi representatives from each state. These endeavors resulted in the curating of a robust Gondi language dataset, containing an impressive 30,000-row sample, and the creation of the ”Gondi Manak Shabdkosh” dictionary. 📚

Preparing the dataset using python
```python
### Get Gondi and Hindi Vocabulary
all_gond_words=set()
for gond in lines['gondi_sentence']:
    for word in gond.split():
        if word not in all_gond_words:
            all_gond_words.add(word)

all_hindi_words=set()
for hin in lines['hindi_sentence']:
    for word in hin.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)
```

## 🎯 Dialect Identification and Mappingg

In our Dialect Identification process, we employ a classifier that incorporates either a supervised machine learning model or a rule-based system. This classifier analyzes input sentences to determine their specific Gondi dialect, relying on unique linguistic features such as syntax, vocabulary usage, and distinct dialectal nuances. Each dialect in our model is represented by a numeric identifier: 1 for Dorla, 2 for Koya, 3 for Madiya, 4 for Muria, and 5 for Raj Gond.

Following Dialect Identification, we implement Dialect-Specific Preprocessing tailored to the characteristics of the identified dialect. This preprocessing involves strategies such as normalizing dialect-specific words and handling unique syntactic structures. This step ensures that the subsequent translation process is optimized for the specific linguistic nuances of each Gondi dialect.

For translation, our Transformer model takes on the task after preprocessing. Specifically designed to handle dialect-specific translation mappings, the model leverages these mappings to enhance translation precision. While integrating Dialect Identification introduces complexity to the preprocessing pipeline, its potential to significantly improve translation accuracy is substantial. This approach tailors the translation process to accommodate dialect-specific idiosyncrasies, refining the quality of translations. However, it's essential to note that successful implementation requires access to labeled data for each dialect and a comprehensive understanding of the distinguishing features of each dialect to create accurate translation mappings.

In UNET's architecture, let's define:

## Transformer Architecture
Our chosen model for the translation task is based on the MarianMT framework, utilizing the Transformer model. The Transformer, initially presented by Vaswani and colleagues in their influential paper "Attention is All You Need," features a non-sequential attention mechanism. In our implementation, this model distinguishes itself by concurrently applying self-attention to the entire sequence, enabling parallel processing and capturing dependencies across distances. This approach proves particularly advantageous in translation tasks, where understanding long-distance relationships between words is crucial. The key components of the Transformer model include encoder and decoder stacks, each consisting of multiple layers.

In our practical application, both the encoder and decoder stacks comprise N layers, with the optimal number determined through experimental evaluation. Within the encoder layers, two sub-layers are employed: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder introduces an additional sub-layer that performs multi-head attention over the encoder stack's output. Residual connections and layer normalization follow these sub-layers. The multi-head attention mechanism, illustrated in Figure 3, allows the model to focus on different positions in the input simultaneously. This configuration, utilizing multiple sets of learned linear transformations, enhances the model's ability to discern and capture intricate relationships within the input data.

Our approach emphasizes the practical implementation of the Transformer model, showcasing its capabilities in addressing challenges posed by sequential data processing and enhancing performance in translation tasks.


<img src = Transform-1.png  width = "800" height = "400">





```python
# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

```


