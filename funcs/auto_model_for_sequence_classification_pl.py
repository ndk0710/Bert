from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from transformers import AutoConfig, AutoModelForSequenceClassification
import torch
from torch import nn
import pytorch_lightning as pl


class AutoModelForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, tokenizer):
        '''
        :parm model_name: Transformerのモデル名
        :parm num_labels: ラベルの数
        :parm lr 学習率
        '''
        super().__init__()

        #num_labelsとlrを保存
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = AutoModelForSequenceClassification.from_config(self.config)
        #トークン追加による変更処理
        '''print(self.bert.base_model.emmbedings.word_embeddings)
        self.bert.base_model.resize_token_embeddings(len(tokenizer))
        print(self.bert.base_model.emmbedings.word_embeddings)'''

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    
    #順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_ooutput)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds, output
    
    #学習データのみにバッチに対して損失出力
    def training_step(self, batch, batch_idx):
        output = self.bert(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.bert(**batch)
        loss = output.loss
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.bert(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)
        self.log('accuracy', accuracy)
    
    def predict_step(self, batch, batch_idx):
        output = self.bert(**batch)
        labels_predicted = output.logits.argmax(-1)
        return labels_predicted
    
    # optimizerの設定
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)
        return [optimizer]