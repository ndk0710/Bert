"""K-Means型クラスタリングのためのコサイン距離カーナル

これは書籍の第4章に付随するオープンソースの例です：
「ヒューマン・イン・ザ・ループ機械学習」

これは一般的なクラスタリング・ライブラリである。

このコードベースでは、3つの能動学習戦略をサポートしている：
1. クラスタベースのサンプリング
2. 代表サンプリング
3. 適応的代表サンプリング


"""
import torch
import torch.nn.functional as F
from random import shuffle

from transformers import AutoTokenizer

def filter_lists(list1, list2):
    # 長さが異なるリストが入力されないようチェック
    if len(list1) != len(list2):
        raise ValueError("リストの長さが同じである必要があります")
    
    # list1の要素が1のときだけlist2の要素をリストに追加
    filtered_list = [b for a, b in zip(list1, list2) if a == 1]
    
    return filtered_list

class CosineClusters():
    """データセット上のクラスタの集合を表す
    
    
    """
    
    
    def __init__(self, cfg, num_clusters=100):
        
        self.clusters = [] # 教師なしおよび軽い教師ありサンプリングのクラスタ
        self.item_cluster = {} # 各アイテムのクラスタを、そのアイテムのidで表す


        # 初期クラスターの作成
        for i in range(0, num_clusters):
            self.clusters.append(Cluster(cfg))
        
        
    def add_random_training_items(self, items):
        """ アイテムをランダムにクラスタに追加する   
        """ 
        
        cur_index = 0
        for item in items:
            self.clusters[cur_index].add_to_cluster(item)
            textid = item[0]
            self.item_cluster[textid] = self.clusters[cur_index]
            
            cur_index += 1
            if cur_index >= len(self.clusters):
                cur_index = 0 


    def add_items_to_best_cluster(self, items):
        """ ベスト・クラスターに複数のアイテムを追加する
        
        """
        added = 0
        for item in items:
            new = self.add_item_to_best_cluster(item)
            if new:
                added += 1
                
        return added



    def get_best_cluster(self, item):
        """ Finds the best cluster for this item
            
            returns the cluster and the score
        """
        best_cluster = None 
        best_fit = float("-inf")        
             
        for cluster in self.clusters:
            fit = cluster.cosine_similary(item)
            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster 
        
        return [best_cluster, best_fit]
    
       

    def add_item_to_best_cluster(self, item):
        """ 最適なクラスターにアイテムを追加  
            
            前のクラスタに存在する場合は、そのクラスタから削除する
            項目が新しいクラスタまたは移動したクラスタであればTrueを返す
            アイテムが同じクラスタに残っている場合はFalseを返す
        """ 
        
        best_cluster = None 
        best_fit = float("-inf")        
        previous_cluster = None
        
        # 現在のクラスターから外し、マッチに貢献しないようにする
        textid = item[0]
        if textid in self.item_cluster:
            previous_cluster = self.item_cluster[textid]
            previous_cluster.remove_from_cluster(item)
            
        for cluster in self.clusters:
            fit = cluster.cosine_similary(item)
            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster 
        
        best_cluster.add_to_cluster(item)
        self.item_cluster[textid] = best_cluster
        
        if best_cluster == previous_cluster:
            return False
        else:
            return True
 
 
    def get_items_cluster(self, item):  
        textid = item[0]
        
        if textid in self.item_cluster:
            return self.item_cluster[textid]
        else:
            return None      
        
        
    def get_centroids(self):  
        centroids = []
        for cluster in self.clusters:
            centroids.append(cluster.get_centroid())
        
        return centroids
    
        
    def get_outliers(self):  
        outliers = []
        for cluster in self.clusters:
            outliers.append(cluster.get_outlier())
        
        return outliers
 
         
    def get_randoms(self, number_per_cluster=1, verbose=False):  
        randoms = []
        for cluster in self.clusters:
            randoms += cluster.get_random_members(number_per_cluster, verbose)
        
        return randoms
   
      
    def shape(self):  
        lengths = []
        for cluster in self.clusters:
            lengths.append(cluster.size())
        
        return str(lengths)



class Cluster():
    """教師なしクラスタリングまたは軽い教師ありクラスタリングでクラスタを表す。
            
    """
    
    feature_idx = {} # 各特徴のインデックスをクラス変数として定数にする


    def __init__(self, cfg):
        self.members = {} # このクラスタ内のアイテムIDによるアイテムのディクショナリ
        self.feature_vector = [] # このクラスタの特徴ベクトル
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name) #トークナイザー
        self.cfg = cfg
    
    def add_to_cluster(self, item):
        textid = item[0]
        text = item[1]
        
        self.members[textid] = item

        encoding = self.tokenizer(
            text,
            max_length=self.cfg.model.max_length,
            padding=self.cfg.model.padding,
            truncation=self.cfg.model.truncation,
        )
        values = encoding.data['input_ids']
        masks = encoding.data['attention_mask']

        words = filter_lists(masks, values)
        #words = text.split()   
        for word in words:
            if word in self.feature_idx:
                while len(self.feature_vector) <= self.feature_idx[word]:
                    self.feature_vector.append(0)
                    
                self.feature_vector[self.feature_idx[word]] += 1
            else:
                # まだどのクラスタにもない新しい特徴                
                self.feature_idx[word] = len(self.feature_vector)
                self.feature_vector.append(1)
                
        
            
    def remove_from_cluster(self, item):
        """ クラスタ内に存在する場合は削除する        
            
        """
        textid = item[0]
        text = item[1]
        
        exists = self.members.pop(textid, False)
        
        if exists:
            encoding = self.tokenizer(
                text,
                max_length=self.cfg.model.max_length,
                padding=self.cfg.model.padding,
                truncation=self.cfg.model.truncation,
            )
            values = encoding.data['input_ids']
            masks = encoding.data['attention_mask']

            words = filter_lists(masks, values)
            #words = text.split()   
            for word in words:
                index = self.feature_idx[word]
                if index < len(self.feature_vector):
                    self.feature_vector[index] -= 1
        
        
    def cosine_similary(self, item):
        text = item[1]

        encoding = self.tokenizer(
            text,
            max_length=self.cfg.model.max_length,
            padding=self.cfg.model.padding,
            truncation=self.cfg.model.truncation,
        )
        values = encoding.data['input_ids']
        masks = encoding.data['attention_mask']

        words = filter_lists(masks, values)
        #words = text.split()  
        
        vector = [0] * len(self.feature_vector)
        for word in words:
            if word not in self.feature_idx:
                self.feature_idx[word] = len(self.feature_vector)
                self.feature_vector.append(0)
                vector.append(1)
            else:
                while len(vector) <= self.feature_idx[word]:
                    vector.append(0)
                    self.feature_vector.append(0)
                              
                vector[self.feature_idx[word]] += 1
        
        item_tensor = torch.FloatTensor(vector)
        cluster_tensor = torch.FloatTensor(self.feature_vector)
        
        similarity = F.cosine_similarity(item_tensor, cluster_tensor, 0)
        
        # F.pairwise_distance()`を使う方法もあるが、最初にクラスタを正規化する
        
        return similarity.item() # item() converts tensor value to float
    
    
    def size(self):
        return len(self.members.keys())
 
 
  
    def get_centroid(self):
        if len(self.members) == 0:
            return []
        
        best_item = None
        best_fit = float("-inf")
        
        for textid in self.members.keys():
            item = self.members[textid]
            similarity = self.cosine_similary(item)
            if similarity > best_fit:
                best_fit = similarity
                best_item = item
        #クラスタリングは終わっているので、リストから除外するのみ
        self.members.pop(best_item[0], False)

        best_item[2] = "cluster_centroid"
        best_item[3] = best_fit

        #best_item[3] = "cluster_centroid"
        #best_item[4] = best_fit 
                
        return best_item
     
         

    def get_outlier(self):
        if len(self.members) == 0:
            return []
        
        best_item = None
        biggest_outlier = float("inf")
        
        for textid in self.members.keys():
            item = self.members[textid]
            similarity = self.cosine_similary(item)
            if similarity < biggest_outlier:
                biggest_outlier = similarity
                best_item = item

        #クラスタリングは終わっているので、リストから除外するのみ
        self.members.pop(best_item[0], False)

        best_item[2] = "cluster_outlier"
        best_item[3] = 1 - biggest_outlier

        #best_item[3] = "cluster_outlier"
        #best_item[4] = 1 - biggest_outlier
                
        return best_item



    def get_random_members(self, number=1, verbose=False):
        if len(self.members) == 0:
            return []        
        
        keys = list(self.members.keys())
        shuffle(keys)

        randoms = []
        for i in range(0, number):
            if i < len(keys):
                textid = keys[i] 
                item = self.members[textid]
                item[2] = "cluster_member"
                item[3] = self.cosine_similary(item)

                #item[3] = "cluster_member"
                #item[4] = self.cosine_similary(item)


                randoms.append(item)
         
        if verbose:
            print("\nRandomly items selected from cluster:")
            for item in randoms:
                print("\t"+item[1])         
                
        return randoms
    




         