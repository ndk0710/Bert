from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
from torch.nn.functional import cross_entropy
#from IPython.display import HTML, display
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
#from funcs.metric import AdaCos
from funcs.auto_model_for_sequence_classification_pl import AutoModelForSequenceClassification_pl
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import top_k_accuracy_score

def calc_top_kaccuracy_score(top_k_labels, top_k_preds, k=2):
    all_probs, all_index, all_count = [], [], []
    for label, pred in zip(top_k_labels, top_k_preds):
        sort_pred = np.sort(pred)[::-1]
        sort_index = np.argsort(pred)[::-1]

        sort_pred = sort_pred[:k]
        sort_index = sort_index[:k]

        if np.any(sort_index == label):
            count = 1
        else:
            count = 0

        all_probs.append(sort_pred)
        all_index.append(sort_index)
        all_count.append(count)
    
    return all_probs, all_index, all_count

def check_feature(losses, features, min_loss, min_feature_vector):
    for index, loss in enumerate(losses):
        if loss < min_loss:
            min_loss = loss
            min_feature_vector = features[index]

    return min_loss, min_feature_vector

#ハイライト処理
def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

#HTML情報作成（可視化）
def mk_html(index, batch, preds, attention_weight, tokenizer, device):
    batch_htmls = []
    for index in range(len(attention_weight)):
        sentence = batch['input_ids'][index].detach()
        label = batch['labels'][index].detach().cpu().numpy()
        pred = preds[index]

        categories = ['EMI', 'MLCC(FMC)', 'MLCC(IMC)', 'TD', 'その他', 'ｻｰﾐｽﾀ']
        id2cat = dict(zip(list(range(len(categories))), categories))
        #cat2id = dict(zip(categories, list(range(len(categories)))))

        label_str = id2cat[int(label)]
        pred_str = id2cat[pred]

        html= '正解カテゴリ: {}<br>予測カテゴリ: {}<br>'.format(label_str, pred_str)

        #文章の長さの分zero tensorを宣言
        seq_len = attention_weight.size()[2]
        all_attens = torch.zeros(seq_len).to(device)

        for i in range(12):
            all_attens += attention_weight[index, i, 0, :]
        
        for word, attn in zip(sentence, all_attens):
            if tokenizer.convert_ids_to_tokens([word.tolist()])[0] == '[SEP]':
                break
            html += '<br><br>'
            batch_htmls.append(html)
            #出力
            with open(f'index_その他_ave_{index+1}.html', 'w') as f:
                f.write(html)
        
    return batch_htmls


def excecute_model(dataloader_pred, cfg, tokenizer):
    device = torch.device('cuda')
    #device = torch.device('cpu')

    model = AutoModelForSequenceClassification.from_pretrained(cfg.directory.use_model_dir).to(device)
    #print(model.bert.base_model.embeddings.word_embeddings)
    torch.set_grad_enabled(False)

    #threshold_value = 0.42

    model.eval()

    all_preds, all_losses, all_features, all_htmls, all_probs, all_indexs, all_counts = [], [], [], [], [], [], []

    min_loss = 10
    min_feature_vector = np.zeros((1, 768))
    with torch.no_grad():
        for _, data in enumerate(dataloader_pred, 0):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)
            #CUDA OUT OF MEMORY
            #preds = model(input_ids, attention_mask, token_type_ids, output_attentions=True)
            preds = model(input_ids, attention_mask, token_type_ids)
            feature = model.bert(input_ids, attention_mask, token_type_ids)
            features = feature['pooler_output'].detach().cpu().numpy()
            all_features.extend(features)
            #print(preds.attentions[-1].size())

            #label_predicted_converted = torch.gt(preds[0], threshold_value) * 1.00
            all_preds.extend(preds[0].argmax(-1).detach().cpu().numpy())

            #top-k-accuracy-score
            top_k_labels = labels.detach().cpu().numpy()
            top_k_preds = F.softmax(preds[0]).detach().cpu().numpy()
            probs, indexs, counts = calc_top_kaccuracy_score(top_k_labels, top_k_preds, k=2)
            all_probs.extend(probs)
            all_indexs.extend(indexs)
            all_counts.extend(counts)

            '''ave_attention = torch.zeros(preds.attention[0].shape).to(device)
            for attention in preds.attentions:
                ave_attention += attention
            ave_attention - ave_attention/12

            html = mk_html(0, data, all_preds, ave_attention, tokenizer, device)'''
            
            #可視化情報取得（最終層のパラメータ）
            '''html = mk_html(0, data, all_preds, preds.attentions[-1].detach(), tokenizer, device)
            all_htmls.extend(html)
            with open('index.html', 'w') as f:
                f.write(html)'''
            
            losses = cross_entropy(preds.logits.detach(), labels, reduction='none').cpu().numpy()
            all_losses.extend(losses)

            min_loss, min_feature_vector = check_feature(losses, features, min_loss, min_feature_vector)

    return all_preds, all_losses, all_features, min_feature_vector, all_probs, all_indexs, all_counts 
