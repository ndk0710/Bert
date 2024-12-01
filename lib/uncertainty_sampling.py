"""
不確実性サンプリング
 
PyTorchでの能動学習のための不確実性サンプリングの例 

これは書籍の第3章に付随するオープンソースの例です：
「Human-in-the-Loop機械学習」

4つの能動学習ストラテジーを含んでいます：
1. 最小信頼度サンプリング
2. 信頼限界サンプリング
3. 信頼度比サンプリング
4. エントロピーに基づくサンプリング


"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import math
from random import shuffle

from transformers import AutoTokenizer

from loading.tokenize_data import tokenize_data
from funcs.create_dataset import create_dataset


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

   

class UncertaintySampling():
    """アクティブ・ラーニングによる不確実性のサンプリング
    
    
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    

    def least_confidence(self, prob_dist, sorted=False):
        """ 
        配列の不確かさスコアを返す
        1が最も不確実である0-1の範囲における最小信頼度サンプリング
        
        確率分布が pytorch テンソルであると仮定する： 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        キーワード引数:
            prob_dist -- 0から1までの実数の総和が1.0になるpytorchテンソル
            sorted -- 確率分布があらかじめ大きいものから小さいものへとソートされているかどうか
        """
        if sorted:
            simple_least_conf = prob_dist.data[0] # 最も自信のある予測
        else:
            simple_least_conf = torch.max(prob_dist) # 最も自信のある予測
                    
        num_labels = prob_dist.numel() # ラベルの数
         
        normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1))
        
        return normalized_least_conf.item()
    
    
    def margin_confidence(self, prob_dist, sorted=False):
        """ 
        Returns the uncertainty score of a probability distribution using
        margin of confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
            
        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True) # sort probs so largest is first
        
        difference = (prob_dist.data[0] - prob_dist.data[1]) # difference between top two props
        margin_conf = 1 - difference 
        
        return margin_conf.item()
        
    
    def ratio_confidence(self, prob_dist, sorted=False):
        """ 
        Returns the uncertainty score of a probability distribution using
        ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        Keyword arguments:
            prob_dist --  pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True) # sort probs so largest is first        
            
        ratio_conf = prob_dist.data[1] / prob_dist.data[0] # ratio between top two props
        
        return ratio_conf.item()
    
    
    def entropy_based(self, prob_dist):
        """ 
        Returns the uncertainty score of a probability distribution using
        entropy 
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
                    
        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        log_probs = prob_dist * torch.log2(prob_dist) # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs)
    
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
        
        return normalized_entropy.item()
        
 
   
    def softmax(self, scores, base=math.e):
        """Returns softmax array for array of scores
        
        Converts a set of raw scores from a model (logits) into a 
        probability distribution via softmax.
            
        The probability distribution will be a set of real numbers
        such that each is in the range 0-1.0 and the sum is 1.0.
    
        Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])
            
        Keyword arguments:
            prediction -- a pytorch tensor of any positive/negative real numbers.
            base -- the base for the exponential (default e)
        """
        exps = (base**scores.to(dtype=torch.float)) # exponential for each value in array
        sum_exps = torch.sum(exps) # sum of all exponentials

        prob_dist = exps / sum_exps # normalize exponentials 
        return prob_dist
        
   
        
        
    def get_samples(self, model, unlabeled_data, device, cfg, number=5, limit=10000):
        """ラベル付けされていないデータから、指定された不確実性サンプリング法でサンプルを取得
    
       キーワード引数:
            model -- このタスクの現在の機械学習モデル
            unlabeled_data -- ラベルのないデータ
            method -- 不確実性サンプリングのメソッド (例: least_confidence())
            feature_method -- データから特徴を抽出する方法
            number -- サンプリングする項目の数
            limit -- 高速サンプリングのために、この数の予測値のみからサンプリングする (-1 = 制限なし)
    
        最小信頼度サンプリングに従って、最も不確実な項目の数を返す
    
        """
    
        samples = []
    
        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose: # we're drawing from *a lot* of data this will take a while                                               
            print("大量のラベルなしデータに対する予測を得る")
        else:
            # 限られたアイテムにのみモデルを適用する                                                                           
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
    
        with torch.no_grad():
            
            #dataloader生成
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

            dataset = []
            text = unlabeled_data
            for j in range(len(unlabeled_data)):
                encoding = tokenizer(
                    text[j],
                    max_length=cfg.model.max_length,
                    padding=cfg.model.padding,
                    truncation=cfg.model.truncation,
                )
                #重要
                encoding = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoding.items()}
                encoding['text'] = text[j]
                dataset.append(encoding)
            
            dataloader = DataLoader(dataset, batch_size=1)

            for index, data in enumerate(dataloader,0):
            #for index, data in enumerate(zip(dataloader, unlabeled_data)):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                preds = model(input_ids, attention_mask, token_type_ids)
                output=preds[0]
                prob_dist = F.softmax(preds[0])
                score = UncertaintySampling.least_confidence(self, prob_dist)
                method = UncertaintySampling.least_confidence.__name__
                samples.append([index, data['text'][0], method, score])
                #samples.append([index, data[1], method, score])
                co=0


            """for item in unlabeled_data:
                text = item[1]
                
                #feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  
    
                prob_dist = torch.exp(log_probs) # the probability distribution of our prediction
                
                score = method(prob_dist.data[0]) # get the specific type of uncertainty sampling
                
                item[3] = method.__name__ # the type of uncertainty sampling used 
                item[4] = score
                
                samples.append(item)"""
                
                
        samples.sort(reverse=True, key=lambda x: x[3])       
        return samples[:number:]        
        
    

        