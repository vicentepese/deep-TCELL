# deep-TCELL
T-cell HLA complex bingind prediction through NLP and Transformers

### Best working model R
Roberta Multi-label:
model_config = RobertaConfig(vocab_size = tokenizer.get_vocab_size(),
                            hidden_size = 1032,
                            num_attention_heads = 12,
                            num_hidden_layers = 12,
                            problem_type="multi_label_classification",
                            hidden_dropout_prob=settings['param']['dropout'])
batchsize=32
learning_rate=1e-5
dropout=0.1-0.3
