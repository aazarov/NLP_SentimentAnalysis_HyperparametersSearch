import numpy as np
import torch
from torch.nn import Module, Embedding, GRU, Linear, Sequential
from torch.nn.utils.rnn import PackedSequence
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


# main Model class: 
# GRU-based bidirectional Encoder 
# with 1 hidden-fully-connected classification output
class Model(Module):
    def __init__(self, lang, gensim_model, config):
        super().__init__() 
        self.config = config
        self.lang = lang
        self.emb_matrix = self.prepare_emb_matrix(gensim_model, self.lang) 
        self.embeddings = Embedding.from_pretrained(self.emb_matrix, freeze=self.config['embedding_freeze'])
        self.gru = GRU(input_size=self.emb_matrix.size(1),
                               batch_first=True,
                               hidden_size=self.config['gru_hidden_size'],
                               num_layers=self.config['gru_num_layers'],
                               dropout=self.config['gru_dropout'],
                               bidirectional=self.config['gru_bidirectional'])      
        
        linear_input_size = self.config['gru_hidden_size'] * \
                            self.config['gru_num_layers'] * \
                            (2 if self.config['gru_bidirectional'] else 1)
        
        out_layers = []
        out_layers.append(Linear(linear_input_size, self.config['fc_size']))            
        out_layers.append(Linear(self.config["fc_size"], self.config['n_classes']))
        self.out_seq = Sequential(*out_layers)        
        
    # Extract embedding matrix from Gensim model for words in Lang.   
    def prepare_emb_matrix(self, gensim_model, lang):
        mean = gensim_model.vectors.mean(1).mean()
        std = gensim_model.vectors.std(1).mean()
        vec_size = gensim_model.vector_size
        emb_matrix = torch.zeros((len(lang), vec_size))
        n_missing_in_gensim=0
        for i in range(len(lang)):
            try:
                emb_matrix[i] = torch.tensor(gensim_model.get_vector(lang.itos[i]))
            except KeyError:
                emb_matrix[i] = torch.randn(vec_size) * std + mean
                n_missing_in_gensim += 1
        #print('n_missing_in_gensim=%i' % n_missing_in_gensim)
        return emb_matrix
        
        
    def forward(self, input):
        embedded = self.embeddings(input.data)
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        _, last_state = self.gru(PackedSequence(embedded,
                                                 input.batch_sizes,
                                                 sorted_indices=input.sorted_indices,
                                                 unsorted_indices=input.unsorted_indices))

        if isinstance(last_state, tuple):
            last_state = last_state[0]
        last_state = last_state.transpose(0, 1)
        last_state = last_state.reshape(last_state.size(0), -1)
        return self.out_seq(last_state)


# Learner class: trains the supplied Model 
# using Adam optimizer
class Learner:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.loss_fn = CrossEntropyLoss()
        self.opt = Adam(self.model.parameters(), 
                        lr=self.config['learning_rate'], 
                        weight_decay=self.config['weight_decay'])
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
    def train_epoch(self, train_loader):
        self.model.train()
        losses = []
        for batch in train_loader:
            self.model.zero_grad()
            texts, labels = batch
            logits = self.model.forward(texts.to(self.device))
            # the CrossEntropyLoss expects raw, un-normalized probabilities for each class
            loss = self.loss_fn(logits, labels.to(self.device))
            loss.backward()
            self.opt.step()
            loss_val = loss.item()
            losses.append(loss_val)
        return np.mean(losses)

    
    def test(self, test_loader):
        self.model.eval()
        losses = []
        num_all = 0
        num_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                self.model.zero_grad()
                texts, labels = batch
                logits = self.model.forward(texts.to(self.device))
                loss = self.loss_fn(logits, labels.to(self.device))
                
                # for accuracy
                preds = torch.argmax(logits, dim=1)
                correct_tensor = preds.eq(labels.float().to(self.device))
                correct = np.squeeze(correct_tensor.cpu().numpy())
                num_correct += np.sum(correct)
                num_all += len(correct)
                
                loss_val = loss.item()
            losses.append(loss_val)

        return np.mean(losses), num_correct / num_all  
        

    def train(self, corpus_holder, budget=None):
        if budget is None:
            train_loader = corpus_holder.train_dataloader
            val_loader = corpus_holder.val_dataloader
        else:
            train_loader = corpus_holder.get_budgeted_train_dataloader(budget)
            val_loader = corpus_holder.get_budgeted_val_dataloader(budget)
        test_loader = corpus_holder.test_dataloader
        if self.config['verbose']:
            print(self.device)
        self.model = self.model.to(self.device)
        for epoch in range(self.config['n_epoch']):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.test(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            if self.config['verbose']:
                print('epoch=%i train_loss=%.4f val_loss=%.4f val_acc=%.4f' % (epoch, train_loss, val_loss, val_acc))
        test_loss, test_acc = self.test(test_loader)
        self.history['test_acc'] = test_acc
        return val_acc, test_acc
    
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        checkpoint = {"model_config": self.model.config,
                      "learner_config": self.config,
                      "lang": self.model.lang,
                      "emb_matrix": self.model.emb_matrix,
                      "state_dict": self.model.state_dict(),
                      "history": self.history}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, gensim_model):
        ckpt = torch.load(path)
        keys = ["model_config", "learner_config", "lang", "emb_matrix", "state_dict", "history"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")
        new_model = Model(ckpt["lang"], gensim_model, ckpt["model_config"])
        new_model.load_state_dict(ckpt["state_dict"])
        new_trainer = cls(new_model, ckpt["learner_config"])
        new_trainer.model.to(new_trainer.device)
        new_trainer.history = ckpt["history"]
        return new_trainer    